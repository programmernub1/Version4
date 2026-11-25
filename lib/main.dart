import 'dart:async';
import 'dart:convert';
import 'dart:io'
    show
        InternetAddress,
        InternetAddressType,
        NetworkInterface,
        RawDatagramSocket,
        RawSocketEvent;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:hand_landmarker/hand_landmarker.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Portrait only: the hand detector assumes an upright camera image.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final state = AppState();
  await state.loadSettings();
  runApp(SteerApp(appState: state));
}

class SteerColors {
  static const Color bg = Color(0xFFF2F7F6);
  static const Color card = Colors.white;
  static const Color teal = Color(0xFF009B7D);
  static const Color tealLight = Color(0xFFE1F3EE);
  static const Color dark = Color(0xFF0F2830);
  static const Color cameraBg = Color(0xFF122832);
  static const Color subtitle = Color(0xFF536B73);
  static const Color danger = Color(0xFFD32F2F);
  static const Color dangerLight = Color(0xFFFCE8E8);
  static const Color warning = Color(0xFFF59E0B);
  static const Color warningYellow = Color(0xFFFFD54F);
}

// ============================================================================
// HAND TRACKING MODEL + GESTURE CLASSIFIER
// ============================================================================
//
// MediaPipe hand landmark order (21 points, normalized 0..1, y points down):
//  0 wrist | 1-4 thumb | 5-8 index | 9-12 middle | 13-16 ring | 17-20 pinky
//
// HOOK: when you add a real MediaPipe/TFLite plugin, call
//   appState.onHandFrame(HandFrame(landmarks))   when a hand is detected
// (and simply stop calling it when no hand is visible - the failsafe
// watchdog in AppState brakes the robot automatically after 500 ms).

// --- tunables for the real MediaPipe tracker (Android) ---
/// GPU is faster but some phones fail with it. If no skeleton ever appears,
/// leave this false (CPU).
const bool kHandUseGpu = false;
const double kHandMinConfidence = 0.6;
/// Palm size limits (fraction of image height) for the "move closer/back" hint.
const double kHandTooFar = 0.10;
const double kHandTooClose = 0.45;

class HandFrame {
  /// 21 landmarks, normalized 0..1 in the UPRIGHT, UNMIRRORED camera image.
  final List<Offset> landmarks;

  /// Upright image width / height (about 0.75 for a 3:4 portrait image).
  final double aspect;
  const HandFrame(this.landmarks, {this.aspect = 0.75});
}

class HandClass {
  final String gesture; // '1'..'4' or 'Palm'
  final int fingers; // raised fingers excluding the thumb (0..4)
  final bool thumbOut;
  const HandClass(this.gesture, this.fingers, this.thumbOut);
}

/// Classifies 21 landmarks. Finger "extended" = the finger is STRAIGHT
/// (angle between knuckle->middle joint and middle joint->tip) and its tip is
/// farther from the wrist than its middle joint. Angle-based, so it keeps
/// working when the hand is rotated or tilted. Anything unclear -> 'Palm'
/// (stop), because stopping is the safe default.
HandClass classifyHandDetailed(
    List<Offset> norm, double aspect, double sensitivity) {
  if (norm.length != 21) return const HandClass('Palm', 0, false);
  // Isotropic units: scale x by the image aspect so angles/distances are true.
  final lm = [for (final p in norm) Offset(p.dx * aspect, p.dy)];
  double d(int a, int b) => (lm[a] - lm[b]).distance;
  final palm = d(0, 9);
  if (palm < 1e-4) return const HandClass('Palm', 0, false);
  final sens = sensitivity.clamp(0.0, 1.0);

  // cos of the bend between (mcp->pip) and (pip->tip): 1 = straight finger.
  double straight(int mcp, int pip, int tip) {
    final a = lm[pip] - lm[mcp];
    final b = lm[tip] - lm[pip];
    final la = a.distance, lb = b.distance;
    if (la < 1e-6 || lb < 1e-6) return -1;
    return (a.dx * b.dx + a.dy * b.dy) / (la * lb);
  }

  // Higher sensitivity -> a less straight finger still counts as raised.
  final thr = 0.80 - 0.40 * sens;
  bool ext(int mcp, int pip, int tip) =>
      straight(mcp, pip, tip) > thr && d(0, tip) > d(0, pip);

  final count = [ext(5, 6, 8), ext(9, 10, 12), ext(13, 14, 16), ext(17, 18, 20)]
      .where((e) => e)
      .length;
  final margin = 0.30 - 0.25 * sens;
  final thumbOut = d(4, 17) > d(3, 17) * (1 + margin * 0.5) && d(4, 5) > 0.45 * palm;

  if (count == 0) return HandClass('Palm', 0, thumbOut); // fist = stop
  if (count == 4 && thumbOut) return const HandClass('Palm', 4, true); // open hand
  return HandClass('$count', count, thumbOut);
}

String classifyHand(List<Offset> lm, double sensitivity, {double aspect = 1.0}) =>
    classifyHandDetailed(lm, aspect, sensitivity).gesture;

const List<List<int>> kHandConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/// Result of the on-device hand detector for one camera frame.
class HandObservation {
  final String gesture; // '1'..'4' or 'Palm'
  final int fingers; // raw finger count (0..5)
  final Offset center; // palm centre, normalized to the upright image (0..1)
  final double radiusNormH; // palm radius as a fraction of image height
  final double aspect; // upright image width / height
  const HandObservation({
    required this.gesture,
    required this.fingers,
    required this.center,
    required this.radiusNormH,
    required this.aspect,
  });
}

int _clampInt(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

/// Lightweight, dependency-free hand detector that runs on the live camera
/// image stream: skin-tone segmentation (YCbCr) -> largest blob -> palm centre
/// (distance transform) -> radial scan to count raised fingers.
///
/// It is a practical stand-in for MediaPipe: it works best with a plain
/// background, good light, spread fingers, thumb tucked for 1-4, and the face
/// outside the guide frame. For production-grade tracking, feed real landmarks
/// through AppState.onHandFrame() instead.
class SkinHandDetector {
  static const int gw = 72; // analysis grid width (cells)
  int _gh = 0;
  Uint8List _mask = Uint8List(0);
  Int32List _queue = Int32List(0);
  Int32List _dt = Int32List(0);

  int _flood(int seed, int from, int to) {
    var head = 0, tail = 0;
    _queue[tail++] = seed;
    _mask[seed] = to;
    while (head < tail) {
      final c = _queue[head++];
      final x = c % gw;
      final y = c ~/ gw;
      if (x > 0 && _mask[c - 1] == from) {
        _mask[c - 1] = to;
        _queue[tail++] = c - 1;
      }
      if (x < gw - 1 && _mask[c + 1] == from) {
        _mask[c + 1] = to;
        _queue[tail++] = c + 1;
      }
      if (y > 0 && _mask[c - gw] == from) {
        _mask[c - gw] = to;
        _queue[tail++] = c - gw;
      }
      if (y < _gh - 1 && _mask[c + gw] == from) {
        _mask[c + gw] = to;
        _queue[tail++] = c + gw;
      }
    }
    return tail;
  }

  HandObservation? detect(
      CameraImage img, int sensorRotation, double sensitivity) {
    final sw = img.width, sh = img.height;
    final rot = ((sensorRotation % 360) + 360) % 360;
    final swap = rot == 90 || rot == 270;
    final uw = swap ? sh : sw; // upright (portrait) image size
    final uh = swap ? sw : sh;
    final gh = (gw * uh / uw).round();
    final n = gw * gh;
    if (_gh != gh) {
      _gh = gh;
      _mask = Uint8List(n);
      _queue = Int32List(n);
      _dt = Int32List(n);
    } else {
      _mask.fillRange(0, n, 0);
    }

    final planes = img.planes;
    final bgra = planes.length == 1;
    if (!bgra && planes.length < 3) return null;
    final p0 = planes[0].bytes;
    final r0 = planes[0].bytesPerRow;
    final pu = bgra ? p0 : planes[1].bytes;
    final pv = bgra ? p0 : planes[2].bytes;
    final ru = bgra ? 0 : planes[1].bytesPerRow;
    final rv = bgra ? 0 : planes[2].bytesPerRow;
    final su = bgra ? 1 : (planes[1].bytesPerPixel ?? 1);
    final sv = bgra ? 1 : (planes[2].bytesPerPixel ?? 1);

    // Region of interest (keeps the face / background out as much as possible).
    final x0 = (gw * 0.12).round(), x1 = (gw * 0.88).round();
    final y0 = (gh * 0.20).round(), y1 = (gh * 0.95).round();
    final roiCells = (x1 - x0) * (y1 - y0);
    var skin = 0;

    for (var gy = y0; gy < y1; gy++) {
      final uy = _clampInt(((gy + 0.5) * uh / gh).floor(), 0, uh - 1);
      for (var gx = x0; gx < x1; gx++) {
        final ux = _clampInt(((gx + 0.5) * uw / gw).floor(), 0, uw - 1);
        int sx, sy;
        switch (rot) {
          case 90:
            sx = uy;
            sy = sh - 1 - ux;
            break;
          case 180:
            sx = sw - 1 - ux;
            sy = sh - 1 - uy;
            break;
          case 270:
            sx = sw - 1 - uy;
            sy = ux;
            break;
          default:
            sx = ux;
            sy = uy;
        }
        int yv, cb, cr;
        if (bgra) {
          final i = sy * r0 + sx * 4;
          if (i + 2 >= p0.length) continue;
          final b = p0[i], g = p0[i + 1], r = p0[i + 2];
          yv = (0.299 * r + 0.587 * g + 0.114 * b).round();
          cb = (128 - 0.168736 * r - 0.331264 * g + 0.5 * b).round();
          cr = (128 + 0.5 * r - 0.418688 * g - 0.081312 * b).round();
        } else {
          final iy = sy * r0 + sx;
          final iu = (sy >> 1) * ru + (sx >> 1) * su;
          final iv = (sy >> 1) * rv + (sx >> 1) * sv;
          if (iy >= p0.length || iu >= pu.length || iv >= pv.length) continue;
          yv = p0[iy];
          cb = pu[iu];
          cr = pv[iv];
        }
        if (yv > 40 && cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173) {
          _mask[gy * gw + gx] = 1;
          skin++;
        }
      }
    }
    if (skin < roiCells * 0.03) return null;

    // Keep only the largest skin blob.
    var bestSeed = -1, bestArea = 0;
    for (var i = 0; i < n; i++) {
      if (_mask[i] != 1) continue;
      final area = _flood(i, 1, 2);
      if (area > bestArea) {
        bestArea = area;
        bestSeed = i;
      }
    }
    if (bestSeed < 0 || bestArea < roiCells * 0.03 || bestArea > roiCells * 0.65) {
      return null; // nothing hand-sized (or something far too big: too close)
    }
    for (var i = 0; i < n; i++) {
      if (_mask[i] == 2) _mask[i] = 1;
    }
    _flood(bestSeed, 1, 3);
    for (var i = 0; i < n; i++) {
      if (_mask[i] == 1) _mask[i] = 0;
    }

    // Chamfer (3-4) distance transform -> palm centre = deepest point.
    const big = 1 << 28;
    for (var y = 0; y < gh; y++) {
      for (var x = 0; x < gw; x++) {
        final i = y * gw + x;
        if (_mask[i] != 3) {
          _dt[i] = 0;
          continue;
        }
        var d = big;
        d = math.min(d, (x > 0 ? _dt[i - 1] : 0) + 3);
        d = math.min(d, (y > 0 ? _dt[i - gw] : 0) + 3);
        d = math.min(d, (x > 0 && y > 0 ? _dt[i - gw - 1] : 0) + 4);
        d = math.min(d, (x < gw - 1 && y > 0 ? _dt[i - gw + 1] : 0) + 4);
        _dt[i] = d;
      }
    }
    var maxD = 0, bestIdx = bestSeed;
    for (var y = gh - 1; y >= 0; y--) {
      for (var x = gw - 1; x >= 0; x--) {
        final i = y * gw + x;
        if (_mask[i] != 3) continue;
        var d = _dt[i];
        d = math.min(d, (x < gw - 1 ? _dt[i + 1] : 0) + 3);
        d = math.min(d, (y < gh - 1 ? _dt[i + gw] : 0) + 3);
        d = math.min(d, (x < gw - 1 && y < gh - 1 ? _dt[i + gw + 1] : 0) + 4);
        d = math.min(d, (x > 0 && y < gh - 1 ? _dt[i + gw - 1] : 0) + 4);
        _dt[i] = d;
        if (d > maxD) {
          maxD = d;
          bestIdx = i;
        }
      }
    }
    final rin = maxD / 3.0;
    if (rin < 3.0) return null;
    final cx = (bestIdx % gw) + 0.5;
    final cy = (bestIdx ~/ gw) + 0.5;

    // Radial scan over the upper arc (+-110 deg from "up"): count protrusions.
    final thr = (1.75 - 0.5 * math.max(0.0, math.min(1.0, sensitivity))) * rin;
    final maxT = math.max(gw, gh).toDouble();
    final samples = <double>[];
    for (var deg = -110; deg <= 110; deg += 3) {
      final a = deg * math.pi / 180.0;
      final dx = math.sin(a), dy = -math.cos(a);
      var last = 0.0;
      var miss = 0;
      for (var t = 0.0; t < maxT; t += 0.5) {
        final ix = (cx + dx * t).floor();
        final iy = (cy + dy * t).floor();
        final inside =
            ix >= 0 && ix < gw && iy >= 0 && iy < gh && _mask[iy * gw + ix] == 3;
        if (inside) {
          last = t;
          miss = 0;
        } else {
          miss++;
          if (miss > 2) break;
        }
      }
      samples.add(last);
    }
    var fingers = 0, run = 0;
    for (var k = 0; k <= samples.length; k++) {
      final above = k < samples.length && samples[k] > thr;
      if (above) {
        run++;
      } else {
        if (run >= 2) fingers += math.max(1, (run / 5.0).round());
        run = 0;
      }
    }

    final gesture = (fingers == 0 || fingers >= 5) ? 'Palm' : '$fingers';
    return HandObservation(
      gesture: gesture,
      fingers: fingers,
      center: Offset(cx / gw, cy / gh),
      radiusNormH: rin / gh,
      aspect: uw / uh,
    );
  }
}

/// Which fingers are raised for a gesture (0 thumb, 1 index ... 4 pinky).
Set<int> extendedFingers(String g) {
  switch (g) {
    case '1':
      return {1};
    case '2':
      return {1, 2};
    case '3':
      return {1, 2, 3};
    case '4':
      return {1, 2, 3, 4};
    case 'Palm':
      return {0, 1, 2, 3, 4};
    default:
      return <int>{};
  }
}

/// 21-point hand skeleton (MediaPipe index order) in palm-radius units,
/// centred on the palm, y pointing down. Raised fingers are extended.
List<Offset> buildHandTemplate(String gesture) {
  final ext = extendedFingers(gesture);
  final pts = <Offset>[const Offset(0, 1.5)]; // 0 wrist
  if (ext.contains(0)) {
    pts.addAll(const [
      Offset(-0.7, 1.0),
      Offset(-1.15, 0.55),
      Offset(-1.55, 0.1),
      Offset(-1.95, -0.3),
    ]);
  } else {
    pts.addAll(const [
      Offset(-0.7, 1.0),
      Offset(-1.0, 0.6),
      Offset(-0.8, 0.2),
      Offset(-0.45, 0.05),
    ]);
  }
  const baseX = [-0.65, -0.2, 0.25, 0.7];
  const baseY = [-0.5, -0.6, -0.5, -0.3];
  const splay = [-0.12, -0.03, 0.06, 0.15];
  const seg = [
    [0.75, 0.55, 0.5],
    [0.85, 0.6, 0.5],
    [0.75, 0.55, 0.5],
    [0.55, 0.4, 0.4],
  ];
  for (var i = 0; i < 4; i++) {
    final base = Offset(baseX[i], baseY[i]);
    pts.add(base);
    if (ext.contains(i + 1)) {
      final raw = Offset(splay[i], -1);
      final dir = raw / raw.distance;
      var cur = base;
      for (var j = 0; j < 3; j++) {
        cur = cur + dir * seg[i][j];
        pts.add(cur);
      }
    } else {
      final pip = base + const Offset(0, -0.35);
      final dip = pip + const Offset(0, 0.25);
      final tip = dip + const Offset(0, 0.2);
      pts.addAll([pip, dip, tip]);
    }
  }
  return pts;
}

// ============================================================================
// APP STATE: network (HTTP), camera, MJPEG, gestures, calibration
// ============================================================================
//
// UDP PACKET FORMAT (ASCII, sent at 20 Hz while connected):
//     "<code>,<left>,<right>"      e.g.  "1,140,140"   "3,-140,140"   "P,0,0"
//   code  : '1' fwd | '2' back | '3' left | '4' right | 'P' palm/stop | 'X' hand lost
//   left/right : signed PWM -255..255 (deadband, speed and trim already applied)
// Heartbeat: app sends "PING" every second; robot should answer "PONG[,battery%]".
// The robot should brake if no packet arrives for ~500 ms.

/// Lifecycle of the phone camera, used to show a spinner / permission prompt.
enum CameraStatus { idle, initializing, ready, denied, error }

/// How hand gestures drive the car.
///  tilt    : fingers = drive (1 fwd, 2 back, fist/palm stop), hand TILT = steer.
///            Driving and steering work at the same time.
///  classic : 1 fwd, 2 back, 3 left, 4 right, palm stop (one thing at a time).
enum ControlMode { tilt, classic }

class AppState extends ChangeNotifier {
  // Network & Hardware State
  String robotIp = '192.168.4.1';
  String udpPort = '8080'; // HTTP control port (ESP32 motorServer)
  String camStreamUrl = 'http://192.168.1.5:81/stream';
  bool isConnected = false;
  bool hasTestedConnection = false;
  bool isScanning = false;
  bool isTesting = false;
  String? statusMessage;
  int linkQuality = 0;
  int batteryLevel = 78;
  int latencyMs = 0;

  /// Demo mode: simulate the robot instead of sending real UDP. Always on for Web.
  bool mockMode = kIsWeb;

  /// Treat the robot as connected even if it never answers PING (one-way UDP).
  bool oneWayMode = false;

  bool get isMock => mockMode || kIsWeb;

  // Camera & Vision State
  bool isFrontCamera = true;
  bool mirrorCamera = true;
  bool showLandmarks = true;
  bool isPipSwapped = false;
  bool isHorizontalSplit = false;
  bool isPracticeMode = false;
  List<Offset>? handLandmarks; // real landmarks, when a tracker feeds us

  // Active Command ('1', '2', '3', '4', 'Palm', 'LOST')
  String activeGesture = 'Palm';
  bool estop = false;

  // Sensitivity & Motor Calibration
  double gestureSensitivity = 0.70;
  double maxMotorSpeed = 0.55;
  double steeringTrim = 0.0; // -10 to +10
  double minPwmDeadband = 0.30; // 0.15 to 0.50 (minimum DRIVE power)

  // Steering (the car has a separate steering motor)
  ControlMode controlMode = ControlMode.tilt;
  double turnStrength = 0.75; // steering power at a full turn, 0.2..1.0
  double turnMin = 0.30; // smallest steering power that moves the wheels
  double turnResponse = 0.50; // 0 = slow / smooth .. 1 = fast
  double tiltFullDeg = 40; // hand tilt that gives a full turn
  double tiltDeadDeg = 12; // tilt smaller than this is ignored
  bool invertSteer = false;

  final ValueNotifier<String> packetNotifier = ValueNotifier('STOP');

  // Live control values, for the on-screen meters.
  // record = (drive, steerPercent, tiltDegrees, speedPercent, turnPercent)
  final ValueNotifier<(int, int, int, int, int)> ctlNotifier =
      ValueNotifier((0, 0, 0, 0, 0));
  double tiltDeg = 0;
  double _tiltSmooth = 0;
  double _tiltSteer = 0;
  double steerOut = 0; // ramped steering actually commanded, -1..1
  int driveOut = 0;

  // Manual (touch) control, active while the Manual screen is open.
  bool manualActive = false;
  int manualDrive = 0;
  double manualSteer = 0;

  // Short steering pulse from the Sensitivity screen's test buttons.
  double _testSteer = 0;
  DateTime _testUntil = DateTime.fromMillisecondsSinceEpoch(0);
  bool get _testActive => DateTime.now().isBefore(_testUntil);
  final ValueNotifier<Uint8List?> robotFrame = ValueNotifier(null);
  bool streamLive = false;

  // --- internals ---
  final math.Random _rng = math.Random();
  RawDatagramSocket? _socket; // unused now (kept so old references compile)
  final http.Client _ctl = http.Client(); // HTTP control client
  String _lastSent = '';
  DateTime _lastSentAt = DateTime.fromMillisecondsSinceEpoch(0);
  InternetAddress? _addr;
  int _port = 8080;
  Completer<void>? _pongWaiter;
  DateTime? _lastPingAt;
  DateTime? _lastPong;
  int _mockTicks = 0;
  Timer? _txTimer;
  Timer? _beatTimer;
  Timer? _watchTimer;

  // tracker debounce / failsafe
  String? _candidate;
  int _candidateCount = 0;
  DateTime _lastHandSeen = DateTime.now();
  bool _trackerLive = false;

  // camera
  CameraController? camController;
  List<CameraDescription> _cams = [];
  String? cameraError;
  CameraStatus cameraStatus = CameraStatus.idle;
  bool _camReinitPending = false;

  // on-device hand tracking (camera image stream -> SkinHandDetector)
  final SkinHandDetector _detector = SkinHandDetector();
  HandObservation? handObs;
  bool trackingActive = false;
  int _trackRefs = 0;
  bool _syncBusy = false;
  bool _syncAgain = false;
  DateTime _lastProc = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _manualHoldUntil = DateTime.fromMillisecondsSinceEpoch(0);
  bool get trackerSupported => !kIsWeb;

  // real MediaPipe tracker (hand_landmarker plugin, Android only)
  HandLandmarkerPlugin? _hl;
  StreamSubscription<List<Hand>>? _hlSub;
  bool _hlFailed = false; // true -> fall back to the skin-colour detector
  int _hlW = 0, _hlH = 0, _hlRot = 90;
  double handAspect = 0.75;
  int fingerCount = 0;
  bool thumbOut = false;
  String observedGesture = 'Palm';
  String handHint = '';
  List<Offset>? _lmSmooth;
  bool get usingMediaPipe => _hl != null && !_hlFailed;

  /// True when the robot camera stream was expected but could not be read.
  bool streamError = false;
  int _camRefs = 0;
  bool _camBusy = false;

  // mjpeg
  int _streamRefs = 0;
  int _streamGen = 0;
  http.Client? _mjpegClient;

  AppState() {
    _txTimer = Timer.periodic(const Duration(milliseconds: 50), (_) => _tick());
    _beatTimer = Timer.periodic(const Duration(seconds: 1), (_) => _beat());
    _watchTimer =
        Timer.periodic(const Duration(milliseconds: 100), (_) => _watch());
  }

  void _notifyLater() => scheduleMicrotask(notifyListeners);

  // ------------------------------------------------------------ persistence
  Future<void> loadSettings() async {
    try {
      final p = await SharedPreferences.getInstance();
      robotIp = p.getString('ip') ?? robotIp;
      udpPort = p.getString('port') ?? udpPort;
      camStreamUrl = p.getString('cam') ?? camStreamUrl;
      gestureSensitivity = p.getDouble('sens') ?? gestureSensitivity;
      maxMotorSpeed = p.getDouble('speed') ?? maxMotorSpeed;
      steeringTrim = p.getDouble('trim') ?? steeringTrim;
      minPwmDeadband = p.getDouble('dead') ?? minPwmDeadband;
      mockMode = kIsWeb || (p.getBool('mock') ?? false);
      oneWayMode = p.getBool('oneway') ?? false;
      turnStrength = p.getDouble('turnStr') ?? turnStrength;
      turnMin = p.getDouble('turnMin') ?? turnMin;
      turnResponse = p.getDouble('turnResp') ?? turnResponse;
      tiltFullDeg = p.getDouble('tiltFull') ?? tiltFullDeg;
      tiltDeadDeg = p.getDouble('tiltDead') ?? tiltDeadDeg;
      invertSteer = p.getBool('invert') ?? invertSteer;
      controlMode = p.getString('mode') == 'classic'
          ? ControlMode.classic
          : ControlMode.tilt;
    } catch (_) {}
  }

  Future<void> saveSettings() async {
    try {
      final p = await SharedPreferences.getInstance();
      await p.setString('ip', robotIp);
      await p.setString('port', udpPort);
      await p.setString('cam', camStreamUrl);
      await p.setDouble('sens', gestureSensitivity);
      await p.setDouble('speed', maxMotorSpeed);
      await p.setDouble('trim', steeringTrim);
      await p.setDouble('dead', minPwmDeadband);
      await p.setBool('mock', mockMode);
      await p.setBool('oneway', oneWayMode);
      await p.setDouble('turnStr', turnStrength);
      await p.setDouble('turnMin', turnMin);
      await p.setDouble('turnResp', turnResponse);
      await p.setDouble('tiltFull', tiltFullDeg);
      await p.setDouble('tiltDead', tiltDeadDeg);
      await p.setBool('invert', invertSteer);
      await p.setString(
          'mode', controlMode == ControlMode.classic ? 'classic' : 'tilt');
    } catch (_) {}
  }

  // ------------------------------------------------------------ commands
  static String nameFor(String g) {
    switch (g) {
      case '1':
        return 'Forward';
      case '2':
        return 'Backward';
      case '3':
        return 'Turn left';
      case '4':
        return 'Turn right';
      case 'Palm':
        return 'Stop';
      case 'LOST':
        return 'Auto-Brake (No Hand)';
      default:
        return 'Stop';
    }
  }

  String get commandName {
    if (estop) return 'Stop';
    if (activeGesture == 'LOST' && !manualActive) return nameFor('LOST');
    final d = driveOut;
    final s = steerOut;
    final side = s.abs() < 0.05 ? '' : (s < 0 ? 'left' : 'right');
    if (d == 0 && side.isEmpty) return 'Stop';
    if (d == 0) return 'Steering $side';
    final dir = d > 0 ? 'Forward' : 'Backward';
    return side.isEmpty ? dir : '$dir, turning $side';
  }

  /// Live hand-tracking feedback line (shown in Drive and Practice mode).
  String get trackingLabel {
    if (!trackerSupported) {
      return 'Hand tracking runs in the Android app (use buttons in browser)';
    }
    if (!trackingActive) {
      return cameraStatus == CameraStatus.ready
          ? 'Starting hand tracking...'
          : 'Waiting for camera...';
    }
    final o = handObs;
    if (handLandmarks != null && usingMediaPipe) {
      final f = fingerCount;
      final hint = handHint.isEmpty ? '' : '  ($handHint)';
      return 'Hand: $f finger${f == 1 ? "" : "s"} -> ${nameFor(observedGesture)}$hint';
    }
    if (o == null) return 'No hand detected';
    final f = o.fingers > 5 ? 5 : o.fingers;
    return 'Hand: $f finger${f == 1 ? "" : "s"} -> ${nameFor(o.gesture)}';
  }

  bool get isMoving =>
      !isPracticeMode && !estop && (driveOut != 0 || steerOut.abs() > 0.05);

  /// Manual gesture selection (on-screen buttons). Also clears an e-stop latch.
  void setGesture(String gesture) {
    estop = false;
    activeGesture = gesture;
    // Let a manual tap win over the camera tracker for a couple of seconds.
    _manualHoldUntil = DateTime.now().add(const Duration(seconds: 2));
    notifyListeners();
  }

  void emergencyStop() {
    estop = true;
    activeGesture = 'Palm';
    manualDrive = 0;
    manualSteer = 0;
    _burstStop();
    notifyListeners();
  }

  /// Silent stop (no notify) - safe to call from dispose().
  void sendStopBurst() {
    activeGesture = 'Palm';
    manualDrive = 0;
    manualSteer = 0;
    _burstStop();
  }

  void onAppPaused() {
    if (isConnected) sendStopBurst();
  }

  void _burstStop() {
    steerOut = 0;
    driveOut = 0;
    _lastSent = '';
    packetNotifier.value = 'STOP';
    ctlNotifier.value = (0, 0, tiltDeg.round(), 0, 0);
    if (isConnected && !isMock) {
      for (var i = 0; i < 3; i++) {
        _sendNow(0, 0, 0, 0);
      }
    }
  }

  /// What the active input wants right now: drive -1/0/1 and steering -1..1
  /// (negative = left, positive = right).
  ({int drive, double steer}) _intent() {
    if (estop) return (drive: 0, steer: 0.0);
    if (_testActive) return (drive: 0, steer: _testSteer);
    if (manualActive) return (drive: manualDrive, steer: manualSteer);
    final g = activeGesture;
    if (g == 'LOST') return (drive: 0, steer: 0.0);

    if (controlMode == ControlMode.classic) {
      switch (g) {
        case '1':
          return (drive: 1, steer: 0.0);
        case '2':
          return (drive: -1, steer: 0.0);
        case '3':
          return (drive: 0, steer: -1.0);
        case '4':
          return (drive: 0, steer: 1.0);
        default:
          return (drive: 0, steer: 0.0);
      }
    }

    // Tilt mode: fingers choose the drive, hand tilt steers. Both at once.
    final drive = g == '1' ? 1 : (g == '2' ? -1 : 0);
    var steer = _tiltSteer;
    if (DateTime.now().difference(_lastHandSeen) >
        const Duration(milliseconds: 300)) {
      steer = 0; // hand not seen very recently: do not steer from old data
    }
    // Latched on-screen LEFT / RIGHT buttons on the Drive screen.
    if (g == '3') steer = -1.0;
    if (g == '4') steer = 1.0;
    return (drive: drive, steer: steer);
  }

  /// Maps the intent to the four numbers sent to the robot:
  /// (drive, steer, speedPercent, turnPercent).
  (int, int, int, int) _toCommand(int drive, double steer) {
    final speed = drive == 0
        ? 0
        : (math.max(maxMotorSpeed, minPwmDeadband) * 100).round();
    final mag = steer.abs();
    if (mag < 0.05) return (drive, 0, speed, 0);
    // Left/right balance: up to 30% more power to one side.
    final t = (steeringTrim / 10.0) * 0.3;
    final side = steer > 0 ? 1 + t : 1 - t;
    final full = (turnStrength * side).clamp(0.1, 1.0).toDouble();
    final low = math.min(turnMin, full);
    final pwm = low + (full - low) * mag;
    final turn = (((pwm * 100) / 5).round() * 5).clamp(0, 100).toInt();
    return (drive, steer > 0 ? 1 : -1, speed, turn);
  }

  void setControlMode(ControlMode m) {
    controlMode = m;
    saveSettings();
    notifyListeners();
  }

  void setInvertSteer(bool v) {
    invertSteer = v;
    _tiltSmooth = 0;
    saveSettings();
    notifyListeners();
  }

  /// Manual (touch) control.
  void setManualActive(bool v) {
    manualActive = v;
    manualDrive = 0;
    manualSteer = 0;
    if (v) isPracticeMode = false;
    _burstStop();
    _notifyLater();
  }

  void setManualDrive(int d) {
    if (d != 0) estop = false;
    manualDrive = d;
    _notifyLater();
  }

  void setManualSteer(double v) {
    if (v != 0) estop = false;
    manualSteer = v;
  }

  /// Short (0.7 s) steering pulse for tuning the turn settings.
  void testSteer(double dir) {
    estop = false;
    _testSteer = dir;
    _testUntil = DateTime.now().add(const Duration(milliseconds: 700));
  }

  // ------------------------------------------------------------ tracker feed
  /// Feed real MediaPipe landmarks (see HandFrame notes).
  void onHandFrame(HandFrame? frame, {List<Offset>? display}) {
    if (frame == null) return; // watchdog handles absence
    _lastHandSeen = DateTime.now();
    _trackerLive = true;
    handObs = null;
    handLandmarks = display ?? frame.landmarks;
    handAspect = frame.aspect;
    _updateTilt(frame);
    final c =
        classifyHandDetailed(frame.landmarks, frame.aspect, gestureSensitivity);
    fingerCount = c.fingers;
    thumbOut = c.thumbOut;
    observedGesture = c.gesture;
    handHint = _hintFor(frame);
    _applyObserved(c.gesture);
    notifyListeners();
  }

  double _steerFromTilt(double deg) {
    final a = deg.abs();
    final full = math.max(tiltFullDeg, tiltDeadDeg + 8);
    if (a <= tiltDeadDeg) return 0;
    final m = ((a - tiltDeadDeg) / (full - tiltDeadDeg)).clamp(0.0, 1.0);
    return deg.sign * m.toDouble();
  }

  /// Hand tilt from the wrist -> middle-finger-knuckle axis. Positive = the
  /// user tilts the top of their hand toward THEIR right.
  void _updateTilt(HandFrame f) {
    final lm = f.landmarks;
    if (lm.length != 21) return;
    final dx = (lm[9].dx - lm[0].dx) * f.aspect;
    final dy = lm[9].dy - lm[0].dy;
    if (dx * dx + dy * dy < 1e-6) return;
    // angle from "up" in the upright, unmirrored image (+ = leans image-right)
    final raw = math.atan2(dx, -dy) * 180 / math.pi;
    // Front camera sees the user from the front: image-right = user's left.
    var user = isFrontCamera ? -raw : raw;
    if (invertSteer) user = -user;
    _tiltSmooth += (user - _tiltSmooth) * 0.45;
    tiltDeg = _tiltSmooth;
    _tiltSteer = _steerFromTilt(_tiltSmooth);
  }

  String _hintFor(HandFrame f) {
    final lm = f.landmarks;
    if (lm.length != 21) return '';
    final dx = (lm[0].dx - lm[9].dx) * f.aspect;
    final dy = lm[0].dy - lm[9].dy;
    final size = math.sqrt(dx * dx + dy * dy);
    final c = lm[9];
    if (c.dx < 0.10 || c.dx > 0.90 || c.dy < 0.10 || c.dy > 0.95) {
      return 'keep your hand inside the frame';
    }
    if (size < kHandTooFar) return 'move your hand closer';
    if (size > kHandTooClose) return 'move your hand back';
    return '';
  }

  /// Called by the built-in skin-colour tracker (fallback) for every frame
  /// with a hand.
  void onHandObservation(HandObservation o) {
    _lastHandSeen = DateTime.now();
    _trackerLive = true;
    handObs = o;
    handLandmarks = null;
    observedGesture = o.gesture;
    handHint = '';
    _tiltSteer = 0; // the fallback detector cannot measure tilt
    _tiltSmooth = 0;
    tiltDeg = 0;
    _applyObserved(o.gesture);
    notifyListeners();
  }

  void _applyObserved(String g) {
    if (DateTime.now().isBefore(_manualHoldUntil)) return;
    // In tilt mode only 1 and 2 fingers drive; 3 or 4 fingers mean stop.
    if (controlMode == ControlMode.tilt && (g == '3' || g == '4')) g = 'Palm';
    // Fast to stop, slow to go: the stop gesture acts at once, while any
    // movement gesture must be held steady for several frames first (this
    // filters the in-between poses while fingers are being raised).
    final need =
        g == 'Palm' ? 1 : (8 - 6 * gestureSensitivity).round().clamp(2, 8);
    if (g == _candidate) {
      _candidateCount++;
    } else {
      _candidate = g;
      _candidateCount = 1;
    }
    if (_candidateCount >= need && g != activeGesture) {
      if (estop && g != 'Palm') return;
      if (g == 'Palm') estop = false;
      activeGesture = g;
    }
  }

  void resetTracker() {
    _trackerLive = false;
    handLandmarks = null;
    handObs = null;
    _candidate = null;
    _candidateCount = 0;
    _lmSmooth = null;
    _tiltSteer = 0;
    _tiltSmooth = 0;
    tiltDeg = 0;
    fingerCount = 0;
    thumbOut = false;
    observedGesture = 'Palm';
    handHint = '';
  }

  // ---- MediaPipe hand landmarker (plugin) ----
  void _startHandLandmarker() {
    if (_hl != null || _hlFailed || !trackerSupported) return;
    try {
      _hl = HandLandmarkerPlugin.create(
        numHands: 1,
        minHandDetectionConfidence: kHandMinConfidence,
        delegate:
            kHandUseGpu ? HandLandmarkerDelegate.gpu : HandLandmarkerDelegate.cpu,
      );
      _hlSub = _hl!.landmarkStream.listen(
        _onHands,
        onError: (_) {
          _hlFailed = true; // use the skin-colour fallback from now on
          _stopHandLandmarker();
        },
      );
    } catch (_) {
      _hlFailed = true;
      _hl = null;
    }
  }

  void _stopHandLandmarker() {
    _hlSub?.cancel();
    _hlSub = null;
    try {
      _hl?.dispose();
    } catch (_) {}
    _hl = null;
    _lmSmooth = null;
  }

  /// Converts a plugin point (normalized in the raw SENSOR image) to the
  /// upright, unmirrored image, using the camera's sensor orientation.
  static Offset _upright(double x, double y, int r) {
    switch (r) {
      case 90:
        return Offset(1 - y, x);
      case 180:
        return Offset(1 - x, 1 - y);
      case 270:
        return Offset(y, 1 - x);
      default:
        return Offset(x, y);
    }
  }

  void _onHands(List<Hand> hands) {
    if (hands.isEmpty || _hlW == 0 || _hlH == 0) {
      if (handLandmarks != null) {
        handLandmarks = null;
        _lmSmooth = null;
        notifyListeners();
      }
      return; // no hand: the 500 ms watchdog brakes the robot
    }
    final raw = hands.first.landmarks;
    if (raw.length != 21) return;
    final r = ((_hlRot % 360) + 360) % 360;
    final swap = r == 90 || r == 270;
    final aspect = swap ? _hlH / _hlW : _hlW / _hlH;
    final pts = [for (final l in raw) _upright(l.x, l.y, r)];

    // Light smoothing for a steadier skeleton (classification uses raw pts).
    final prev = _lmSmooth;
    final sm = prev == null
        ? pts
        : [for (var i = 0; i < 21; i++) prev[i] + (pts[i] - prev[i]) * 0.6];
    _lmSmooth = sm;
    onHandFrame(HandFrame(pts, aspect: aspect), display: sm);
  }

  void _watch() {
    // Hand-lost failsafe: tracker was live but silent for >500 ms.
    if (_trackerLive &&
        activeGesture != 'LOST' &&
        !DateTime.now().isBefore(_manualHoldUntil) &&
        DateTime.now().difference(_lastHandSeen) >
            const Duration(milliseconds: 500)) {
      activeGesture = 'LOST';
      handLandmarks = null;
      _tiltSteer = 0;
      _tiltSmooth = 0;
      tiltDeg = 0;
      notifyListeners();
    }
  }

  void _onCameraImage(CameraImage img) {
    final now = DateTime.now();
    if (now.difference(_lastProc) < const Duration(milliseconds: 50)) return;
    _lastProc = now;
    final rot = camController?.description.sensorOrientation ?? 90;

    // Preferred: real MediaPipe hand landmarks (results arrive in _onHands).
    if (_hl != null && !_hlFailed) {
      try {
        _hlW = img.width;
        _hlH = img.height;
        _hlRot = rot;
        _hl!.processFrame(img, rot);
        return;
      } catch (_) {
        _hlFailed = true;
        _stopHandLandmarker();
      }
    }

    // Fallback: skin-colour detector.
    try {
      final obs = _detector.detect(img, rot, gestureSensitivity);
      if (obs != null) {
        onHandObservation(obs);
      } else if (handObs != null) {
        handObs = null;
        notifyListeners();
      }
    } catch (_) {}
  }

  // ------------------------------------------------------------ UDP transport
  void _tick() {
    final it = _intent();

    // Ramp steering toward the target: smooth, proportional steering.
    // Releasing is twice as fast as turning in.
    final target = it.steer;
    final rate = 0.06 + 0.54 * turnResponse;
    final step = target.abs() < steerOut.abs() ? rate * 2 : rate;
    steerOut += (target - steerOut).clamp(-step, step).toDouble();
    if (target == 0 && steerOut.abs() < 0.05) steerOut = 0;
    if (estop || (activeGesture == 'LOST' && !manualActive && !_testActive)) {
      steerOut = 0; // safety: no hand / e-stop -> straight at once
    }
    driveOut = estop ? 0 : it.drive;

    final preview = _toCommand(driveOut, steerOut);
    ctlNotifier.value = (
      driveOut,
      (steerOut * 100).round(),
      tiltDeg.round(),
      preview.$3,
      preview.$4,
    );

    final live = !isPracticeMode || _testActive;
    final cmd = live ? preview : (0, 0, 0, 0);
    packetNotifier.value =
        'drive=${cmd.$1} speed=${cmd.$3} steer=${cmd.$2} turn=${cmd.$4}';

    if (!isConnected || !live || isMock) return;
    _sendCmd(cmd);
  }

  void _beat() {
    if (isMock) {
      if (isConnected) {
        latencyMs = 40 + _rng.nextInt(50);
        linkQuality = 88 + _rng.nextInt(9);
        _mockTicks++;
        if (_mockTicks % 30 == 0 && batteryLevel > 5) batteryLevel--;
        notifyListeners();
      }
      return;
    }
    if (_addr == null) return;
    if (isConnected || oneWayMode) {
      _lastPingAt = DateTime.now();
      unawaited(_httpPing());
    }
    if (isConnected &&
        !oneWayMode &&
        _lastPong != null &&
        DateTime.now().difference(_lastPong!) > const Duration(seconds: 4)) {
      isConnected = false;
      linkQuality = 0;
      statusMessage = 'Link lost - no reply from robot';
      notifyListeners();
    }
  }

  /// HTTP endpoint the ESP32 firmware must serve.
  /// Request: GET http://<ip>:<port>/move?drive=-1|0|1&steer=-1|0|1&speed=0..100&turn=0..100
  /// Health:  GET http://<ip>:<port>/status   (any HTTP reply counts as alive)
  static const String kMovePath = '/move';
  static const String kStatusPath = '/status';

  /// Sends on change (at most every 90 ms), otherwise repeats every 300 ms as
  /// a heartbeat so the robot's own timeout never trips while driving.
  /// A change to "stopped" is sent immediately.
  void _sendCmd((int, int, int, int) c) {
    if (_addr == null) return;
    final key = '${c.$1},${c.$2},${c.$3},${c.$4}';
    final now = DateTime.now();
    final since = now.difference(_lastSentAt);
    final isStop = c.$1 == 0 && c.$2 == 0;
    final changed = key != _lastSent;
    final due = changed
        ? (isStop || since >= const Duration(milliseconds: 90))
        : since >= const Duration(milliseconds: 300);
    if (!due) return;
    _lastSent = key;
    _lastSentAt = now;
    _sendNow(c.$1, c.$2, c.$3, c.$4);
  }

  void _sendNow(int drive, int steer, int speed, int turn) {
    if (_addr == null) return;
    final uri = Uri.parse('http://$robotIp:$_port$kMovePath').replace(
      queryParameters: {
        'drive': '$drive',
        'steer': '$steer',
        'speed': '$speed',
        'turn': '$turn',
      },
    );
    unawaited(_ctl
        .get(uri)
        .timeout(const Duration(milliseconds: 800))
        .then<void>((_) {})
        .catchError((_) {}));
  }

  Future<void> _httpPing() async {
    try {
      final sw = Stopwatch()..start();
      await _ctl
          .get(Uri.parse('http://$robotIp:$_port$kStatusPath'))
          .timeout(const Duration(milliseconds: 1500));
      latencyMs = sw.elapsedMilliseconds;
      linkQuality = (100 - latencyMs ~/ 4).clamp(5, 100);
      _lastPong = DateTime.now();
      notifyListeners();
    } catch (_) {}
  }

  /// Resolves the target address/port (no socket is needed for HTTP).
  Future<bool> _openSocket() async {
    if (kIsWeb) return false;
    final addr = InternetAddress.tryParse(robotIp);
    final port = int.tryParse(udpPort);
    if (addr == null || port == null || port < 1 || port > 65535) {
      statusMessage = 'Invalid IP address or port';
      return false;
    }
    _addr = addr;
    _port = port;
    return true;
  }

  Future<bool> _handshake() async {
    if (!await _openSocket()) return false;
    try {
      final sw = Stopwatch()..start();
      // Any HTTP reply, even a 404, proves the ESP32 is reachable.
      await _ctl
          .get(Uri.parse('http://$robotIp:$_port$kStatusPath'))
          .timeout(const Duration(seconds: 2));
      latencyMs = sw.elapsedMilliseconds;
      linkQuality = (100 - latencyMs ~/ 4).clamp(5, 100);
      _lastPong = DateTime.now();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<void> testConnection(String ip, String port, String camUrl) async {
    robotIp = ip.trim();
    udpPort = port.trim().isEmpty ? '8080' : port.trim();
    camStreamUrl = camUrl.trim();
    hasTestedConnection = true;
    isTesting = true;
    statusMessage = null;
    notifyListeners();
    restartStream();

    if (isMock) {
      await Future.delayed(const Duration(milliseconds: 450));
      isConnected = robotIp.isNotEmpty;
      latencyMs = 40 + _rng.nextInt(50);
      linkQuality = 90;
      if (!isConnected) statusMessage = 'Camera-only demo (no robot address)';
    } else if (robotIp.isEmpty) {
      isConnected = false;
      statusMessage = 'Camera-only demo (no robot address)';
    } else {
      final ok = await _handshake();
      isConnected = ok || (oneWayMode && _addr != null);
      if (!ok) {
        statusMessage = oneWayMode
            ? 'No reply - sending one-way'
            : 'No reply from robot. Check IP, port and that phone and ESP32 are on the same Wi-Fi';
      }
    }
    isTesting = false;
    saveSettings();
    notifyListeners();
  }

  Future<void> scanNetwork() async {
    isScanning = true;
    statusMessage = null;
    notifyListeners();
    if (isMock) {
      await Future.delayed(const Duration(milliseconds: 900));
      robotIp = '192.168.4.1';
      camStreamUrl = 'http://192.168.4.2:81/stream';
      isConnected = true;
      hasTestedConnection = true;
    } else {
      final found = await _scanLan();
      hasTestedConnection = true;
      if (found != null) {
        robotIp = found;
        isConnected = await _handshake();
      } else {
        statusMessage ??= 'No robot found. Type the ESP32 IP.';
      }
    }
    isScanning = false;
    saveSettings();
    notifyListeners();
  }

  Future<String?> _scanLan() async {
    // HTTP probing would also hit routers and printers, so auto-scan is off.
    // Type the IP printed in the Arduino Serial Monitor instead.
    statusMessage = 'Auto-Scan is not available over HTTP. Type the ESP32 IP.';
    return null;
  }

  void setMockMode(bool v) {
    mockMode = kIsWeb || v;
    isConnected = false;
    statusMessage = null;
    saveSettings();
    notifyListeners();
  }

  void setOneWayMode(bool v) {
    oneWayMode = v;
    saveSettings();
    notifyListeners();
  }

  // ------------------------------------------------------------ camera
  Future<void> acquireCamera() async {
    _camRefs++;
    if (camController != null || _camBusy) return;
    await _initCamera();
  }

  void releaseCamera() {
    _camRefs = math.max(0, _camRefs - 1);
    if (_camRefs == 0) {
      final c = camController;
      camController = null;
      cameraStatus = CameraStatus.idle;
      _notifyLater();
      Future.delayed(const Duration(milliseconds: 150), () => _disposeController(c));
    }
  }

  /// Initializes the physical camera for the currently selected lens.
  /// Requests camera permission (the plugin shows the system / browser prompt)
  /// and records a status the UI uses for the spinner or permission message.
  Future<void> _initCamera() async {
    _camBusy = true;
    cameraStatus = CameraStatus.initializing;
    cameraError = null;
    _notifyLater(); // may run during initState, so never notify synchronously
    try {
      if (_cams.isEmpty) _cams = await availableCameras();
      if (_cams.isEmpty) throw StateError('no cameras');
      final want =
          isFrontCamera ? CameraLensDirection.front : CameraLensDirection.back;
      final desc = _cams.firstWhere(
        (c) => c.lensDirection == want,
        orElse: () => _cams.first,
      );
      // Keep the UI honest if the device lacks the requested lens.
      isFrontCamera = desc.lensDirection == CameraLensDirection.front;
      final fmt = defaultTargetPlatform == TargetPlatform.iOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420;
      CameraController? ready;
      Object? lastErr;
      // Attempt 1: medium + YUV (needed for hand tracking).
      // Attempt 2: low + plugin defaults, for devices that reject attempt 1.
      for (var attempt = 0; attempt < 2 && ready == null; attempt++) {
        final cc = CameraController(
          desc,
          attempt == 0 ? ResolutionPreset.medium : ResolutionPreset.low,
          enableAudio: false,
          imageFormatGroup: attempt == 0 ? fmt : null,
        );
        try {
          await cc.initialize(); // triggers the permission prompt on first use
          ready = cc;
        } on CameraException catch (e) {
          lastErr = e;
          try {
            await cc.dispose();
          } catch (_) {}
          if (_isDenied(e)) break;
        }
      }
      final c = ready;
      if (c == null) throw lastErr ?? StateError('camera init failed');
      if (_camRefs == 0) {
        await c.dispose();
        cameraStatus = CameraStatus.idle;
        return;
      }
      camController = c;
      cameraStatus = CameraStatus.ready;
    } on CameraException catch (e) {
      camController = null;
      final denied = _isDenied(e);
      cameraStatus = denied ? CameraStatus.denied : CameraStatus.error;
      cameraError = denied
          ? (kIsWeb
              ? 'Camera blocked (${e.code}). Allow it in the browser address bar; browsers need https or localhost.'
              : 'Camera permission denied (${e.code}). If no prompt appeared, the CAMERA permission is missing from AndroidManifest.xml (run tools/patch_android.py and reinstall).')
          : '${e.code}: ${e.description ?? "camera error"}';
    } catch (e) {
      camController = null;
      cameraStatus = CameraStatus.error;
      cameraError = 'Camera error: $e';
    } finally {
      _camBusy = false;
      notifyListeners();
      unawaited(_syncTracking());
      if (_camReinitPending && _camRefs > 0) {
        _camReinitPending = false;
        final old = camController;
        camController = null;
        await _disposeController(old);
        await _initCamera();
      }
    }
  }

  static bool _isDenied(CameraException e) =>
      e.code.startsWith('CameraAccessDenied') ||
      e.code == 'CameraAccessRestricted' ||
      e.code == 'cameraPermission';

  /// Switches between the physical front and rear lenses.
  Future<void> toggleFrontCamera() async {
    isFrontCamera = !isFrontCamera;
    notifyListeners();
    if (_camRefs == 0) return;
    if (_camBusy) {
      _camReinitPending = true; // re-run with the new lens when init finishes
      return;
    }
    final old = camController;
    camController = null;
    cameraStatus = CameraStatus.initializing;
    notifyListeners();
    await _disposeController(old);
    await _initCamera();
  }

  /// "Retry" button on the permission / error panel.
  Future<void> retryCamera() async {
    if (_camBusy || _camRefs == 0) return;
    final old = camController;
    camController = null;
    await _disposeController(old);
    await _initCamera();
  }

  /// Release the camera while the app is in the background (plugin guidance).
  void onAppBackgrounded() {
    final c = camController;
    if (c == null) return;
    camController = null;
    cameraStatus = CameraStatus.idle;
    trackingActive = false;
    handObs = null;
    notifyListeners();
    Future.delayed(const Duration(milliseconds: 100), () => _disposeController(c));
  }

  Future<void> _disposeController(CameraController? c) async {
    if (c == null) return;
    trackingActive = false;
    _stopHandLandmarker();
    try {
      if (c.value.isStreamingImages) await c.stopImageStream();
    } catch (_) {}
    try {
      await c.dispose();
    } catch (_) {}
  }

  // --- hand tracking on the camera image stream (Drive + Practice) ---
  void acquireTracking() {
    _trackRefs++;
    unawaited(_syncTracking());
  }

  void releaseTracking() {
    _trackRefs = math.max(0, _trackRefs - 1);
    if (_trackRefs == 0) resetTracker();
    unawaited(_syncTracking());
  }

  Future<void> _syncTracking() async {
    if (_syncBusy) {
      _syncAgain = true;
      return;
    }
    _syncBusy = true;
    try {
      do {
        _syncAgain = false;
        await _syncOnce();
      } while (_syncAgain);
    } finally {
      _syncBusy = false;
    }
  }

  Future<void> _syncOnce() async {
    final c = camController;
    if (c == null) {
      trackingActive = false;
      _stopHandLandmarker();
      return;
    }
    final want = _trackRefs > 0 && trackerSupported && c.value.isInitialized;
    final streaming = c.value.isStreamingImages;
    if (want && !streaming) {
      try {
        _startHandLandmarker();
        await c.startImageStream(_onCameraImage);
        trackingActive = true;
        _trackerLive = true; // from now on, no hand for 500 ms => auto-brake
        _lastHandSeen = DateTime.now();
      } catch (_) {
        trackingActive = false;
      }
      _notifyLater();
    } else if (!want && streaming) {
      try {
        await c.stopImageStream();
      } catch (_) {}
      trackingActive = false;
      _stopHandLandmarker();
      _notifyLater();
    }
  }

  /// Re-open the camera on resume (also picks up permission granted in Settings).
  void onAppResumed() {
    if (_camRefs > 0 && camController == null && !_camBusy) _initCamera();
  }

  // ------------------------------------------------------------ MJPEG stream
  void acquireStream() {
    _streamRefs++;
    if (_streamRefs == 1) _runStream();
  }

  void releaseStream() {
    _streamRefs = math.max(0, _streamRefs - 1);
    if (_streamRefs == 0) _stopStream();
  }

  void restartStream() {
    if (_streamRefs > 0) {
      _stopStream();
      _runStream();
    }
  }

  void _stopStream() {
    _streamGen++;
    _mjpegClient?.close();
    _mjpegClient = null;
    streamLive = false;
    streamError = false;
    scheduleMicrotask(() {
      robotFrame.value = null;
      notifyListeners();
    });
  }

  static int _find(List<int> b, int a, int c, int from) {
    for (var i = from; i < b.length - 1; i++) {
      if (b[i] == a && b[i + 1] == c) return i;
    }
    return -1;
  }

  /// Reads the ESP32-CAM MJPEG stream, but only while a real ESP32 robot is
  /// connected. While offline / in demo mode no request is made.
  Future<void> _runStream() async {
    final gen = ++_streamGen;
    while (gen == _streamGen) {
      final robotOnline = isConnected && !isMock;
      final uri = Uri.tryParse(camStreamUrl);
      if (!robotOnline ||
          camStreamUrl.isEmpty ||
          uri == null ||
          !uri.hasScheme) {
        final err = robotOnline; // online, but no usable stream address
        if (robotFrame.value != null || streamLive || streamError != err) {
          robotFrame.value = null;
          streamLive = false;
          streamError = err;
          notifyListeners();
        }
        await Future.delayed(const Duration(milliseconds: 700));
        continue;
      }

      final client = http.Client();
      _mjpegClient = client;
      final buf = <int>[];
      try {
        final resp = await client
            .send(http.Request('GET', uri))
            .timeout(const Duration(seconds: 5));
        if (resp.statusCode != 200) throw StateError('HTTP ${resp.statusCode}');
        await for (final chunk in resp.stream) {
          if (gen != _streamGen || !isConnected) break;
          buf.addAll(chunk);
          while (true) {
            final s = _find(buf, 0xFF, 0xD8, 0);
            if (s < 0) {
              if (buf.length > 4) buf.removeRange(0, buf.length - 1);
              break;
            }
            final e = _find(buf, 0xFF, 0xD9, s + 2);
            if (e < 0) {
              if (s > 0) buf.removeRange(0, s);
              break;
            }
            robotFrame.value = Uint8List.fromList(buf.sublist(s, e + 2));
            buf.removeRange(0, e + 2);
            if (!streamLive || streamError) {
              streamLive = true;
              streamError = false;
              notifyListeners();
            }
          }
          if (buf.length > 2 * 1024 * 1024) buf.clear();
        }
      } catch (_) {
        // unreachable / refused / bad status -> reported as "No camera detected!"
      } finally {
        client.close();
      }
      if (gen != _streamGen) break;
      streamLive = false;
      robotFrame.value = null;
      if (isConnected && !isMock) streamError = true;
      notifyListeners();
      await Future.delayed(const Duration(seconds: 2));
    }
  }

  // ------------------------------------------------------------ view toggles
  void toggleMirror() {
    mirrorCamera = !mirrorCamera;
    notifyListeners();
  }

  void toggleLandmarks() {
    showLandmarks = !showLandmarks;
    notifyListeners();
  }

  void togglePipSwap() {
    isPipSwapped = !isPipSwapped;
    notifyListeners();
  }

  void toggleSplitView() {
    isHorizontalSplit = !isHorizontalSplit;
    notifyListeners();
  }

  void setPracticeMode(bool value) {
    isPracticeMode = value;
    if (value) _burstStop(); // make sure the robot is stopped before practising
    notifyListeners();
  }

  void resetSensitivityDefaults() {
    gestureSensitivity = 0.70;
    maxMotorSpeed = 0.55;
    steeringTrim = 0.0;
    minPwmDeadband = 0.30;
    turnStrength = 0.75;
    turnMin = 0.30;
    turnResponse = 0.50;
    tiltFullDeg = 40;
    tiltDeadDeg = 12;
    invertSteer = false;
    notifyListeners();
  }

  void updateSensitivity({
    double? gesture,
    double? speed,
    double? trim,
    double? deadband,
    double? turn,
    double? turnMinimum,
    double? response,
    double? tiltFull,
    double? tiltDead,
  }) {
    if (gesture != null) gestureSensitivity = gesture;
    if (speed != null) maxMotorSpeed = speed;
    if (trim != null) steeringTrim = trim;
    if (deadband != null) minPwmDeadband = deadband;
    if (turn != null) turnStrength = turn;
    if (turnMinimum != null) turnMin = turnMinimum;
    if (response != null) turnResponse = response;
    if (tiltFull != null) tiltFullDeg = tiltFull;
    if (tiltDead != null) tiltDeadDeg = tiltDead;
    // Keep the dead zone clearly smaller than the full-turn angle.
    if (tiltFullDeg < tiltDeadDeg + 8) {
      if (tiltFull != null) {
        tiltDeadDeg = tiltFullDeg - 8;
      } else {
        tiltFullDeg = tiltDeadDeg + 8;
      }
    }
    notifyListeners();
  }

  @override
  void dispose() {
    _txTimer?.cancel();
    _beatTimer?.cancel();
    _watchTimer?.cancel();
    _stopStream();
    _socket?.close();
    _ctl.close();
    _disposeController(camController);
    super.dispose();
  }
}

// ============================================================================
// APP ROOT
// ============================================================================

class SteerApp extends StatefulWidget {
  final AppState appState;
  const SteerApp({super.key, required this.appState});

  @override
  State<SteerApp> createState() => _SteerAppState();
}

class _SteerAppState extends State<SteerApp> with WidgetsBindingObserver {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    widget.appState.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Never leave the robot driving when the app is backgrounded.
    if (state == AppLifecycleState.resumed) {
      widget.appState.onAppResumed();
    } else {
      widget.appState.onAppPaused();
      if (state == AppLifecycleState.paused) {
        widget.appState.onAppBackgrounded();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'S.T.E.E.R.',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        scaffoldBackgroundColor: SteerColors.bg,
        colorScheme: ColorScheme.fromSeed(seedColor: SteerColors.teal),
        useMaterial3: true,
      ),
      home: OnboardingScreen(appState: widget.appState),
    );
  }
}

// ============================================================================
// SHARED WIDGETS
// ============================================================================

class SteerBrandHeader extends StatelessWidget {
  final AppState appState;
  const SteerBrandHeader({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final on = appState.isConnected;
        return Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: SteerColors.teal,
                borderRadius: BorderRadius.circular(11),
              ),
              alignment: Alignment.center,
              child: const Text(
                'S',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w800,
                  fontSize: 18,
                ),
              ),
            ),
            const SizedBox(width: 10),
            const Text(
              'S.T.E.E.R.',
              style: TextStyle(
                color: SteerColors.dark,
                fontWeight: FontWeight.w800,
                fontSize: 18,
                letterSpacing: 0.5,
              ),
            ),
            const Spacer(),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: on ? SteerColors.tealLight : Colors.white,
                borderRadius: BorderRadius.circular(20),
                border:
                    Border.all(color: on ? SteerColors.teal : Colors.black12),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      color: on ? SteerColors.teal : Colors.grey,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    on
                        ? (appState.isMock ? 'Demo Robot Online' : 'ESP32 Online')
                        : 'Demo / Offline',
                    style: const TextStyle(
                      color: SteerColors.dark,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }
}

class SteerPageTopBar extends StatelessWidget {
  final String title;
  final AppState appState;
  final Widget? trailing;

  const SteerPageTopBar({
    super.key,
    required this.title,
    required this.appState,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            InkWell(
              onTap: () => Navigator.of(context).pop(),
              borderRadius: BorderRadius.circular(14),
              child: Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                ),
                alignment: Alignment.center,
                child: const Icon(
                  Icons.chevron_left_rounded,
                  color: SteerColors.dark,
                  size: 26,
                ),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  color: SteerColors.dark,
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            if (trailing != null) trailing!,
          ],
        ),
        const SizedBox(height: 8),
        ConnectionPill(appState: appState),
      ],
    );
  }
}

/// Connection status chip (Demo / Offline, Demo Robot Online, ESP32 Online).
/// Shown on every page.
class ConnectionPill extends StatelessWidget {
  final AppState appState;
  const ConnectionPill({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final on = appState.isConnected;
        final label = on
            ? (appState.isMock
                ? 'Demo Robot Online'
                : 'ESP32 Online  ${appState.latencyMs} ms')
            : 'Demo / Offline';
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            color: on ? SteerColors.tealLight : Colors.white,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: on ? SteerColors.teal : Colors.black12),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(
                  color: on ? SteerColors.teal : Colors.grey,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                label,
                style: const TextStyle(
                  color: SteerColors.dark,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

IconData gestureIconFor(String code) {
  switch (code) {
    case '1':
      return Icons.touch_app_rounded;
    case '2':
      return Icons.pan_tool_alt_rounded;
    case '3':
      return Icons.back_hand_outlined;
    case '4':
      return Icons.front_hand_outlined;
    case 'Palm':
      return Icons.pan_tool_rounded;
    default:
      return Icons.warning_amber_rounded;
  }
}

/// Live phone camera (hand camera). Shows a spinner / permission prompt until
/// the hardware is ready and paints the 21-point hand skeleton on top:
///  - real plugin landmarks if AppState.handLandmarks is set,
///  - otherwise a skeleton anchored on the hand found by the built-in tracker,
///  - otherwise (only if [simulateIfIdle]) a gesture-driven demo skeleton.
class HandCameraView extends StatelessWidget {
  final AppState appState;
  final bool showOverlay;
  final bool simulateIfIdle;
  const HandCameraView({
    super.key,
    required this.appState,
    this.showOverlay = true,
    this.simulateIfIdle = false,
  });

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final c = appState.camController;
        // The plugin already mirrors the front camera preview, so we only
        // flip when the user's mirror setting differs from that default.
        final flip = appState.mirrorCamera != appState.isFrontCamera;
        final children = <Widget>[];

        if (c != null && c.value.isInitialized) {
          final portrait =
              MediaQuery.of(context).orientation == Orientation.portrait;
          final ar = c.value.aspectRatio;
          final w = portrait ? 100.0 : 100.0 * ar;
          final h = portrait ? 100.0 * ar : 100.0;
          Widget p = FittedBox(
            fit: BoxFit.cover,
            clipBehavior: Clip.hardEdge,
            child: SizedBox(width: w, height: h, child: CameraPreview(c)),
          );
          if (flip) {
            p = Transform(
              alignment: Alignment.center,
              transform: Matrix4.diagonal3Values(-1, 1, 1),
              child: p,
            );
          }
          children.add(Positioned.fill(child: ClipRect(child: p)));
        } else {
          children.add(Positioned.fill(
            child: CameraStatusPanel(
              status: appState.cameraStatus,
              message: appState.cameraError,
              onRetry: appState.retryCamera,
            ),
          ));
        }

        if (showOverlay && appState.showLandmarks) {
          final lm = appState.handLandmarks;
          final obs = appState.handObs;
          if (lm != null) {
            children.add(Positioned.fill(
              child: CustomPaint(
                painter: HandLandmarksPainter(
                  landmarks: lm,
                  aspect: appState.handAspect,
                  mirror: appState.mirrorCamera,
                ),
              ),
            ));
          } else if (obs != null) {
            children.add(Positioned.fill(
              child: CustomPaint(
                painter: HandOverlayPainter(
                  template: buildHandTemplate(obs.gesture),
                  center: obs.center,
                  radiusNormH: obs.radiusNormH,
                  aspect: obs.aspect,
                  mirror: appState.mirrorCamera,
                ),
              ),
            ));
          } else if (simulateIfIdle &&
              !appState.trackingActive &&
              appState.activeGesture != 'LOST') {
            children.add(Positioned.fill(
              child: CustomPaint(
                painter: HandOverlayPainter(
                  template: buildHandTemplate(appState.activeGesture),
                  mirror: appState.mirrorCamera,
                ),
              ),
            ));
          }
        }
        return Stack(fit: StackFit.expand, children: children);
      },
    );
  }
}

/// Spinner while the camera starts, or a permission / error prompt with Retry.
class CameraStatusPanel extends StatelessWidget {
  final CameraStatus status;
  final String? message;
  final VoidCallback onRetry;
  const CameraStatusPanel({
    super.key,
    required this.status,
    required this.message,
    required this.onRetry,
  });

  @override
  Widget build(BuildContext context) {
    final loading =
        status == CameraStatus.idle || status == CameraStatus.initializing;
    final denied = status == CameraStatus.denied;

    final title = loading
        ? 'Starting camera...'
        : denied
            ? 'Camera permission needed'
            : 'Camera unavailable';
    final body = message ??
        (denied
            ? 'Allow camera access for S.T.E.E.R. when prompted, or enable it in your phone/browser settings, then tap Retry.'
            : 'Showing simulated view.');

    return LayoutBuilder(
      builder: (context, c) {
        final compact = c.maxHeight < 220 || c.maxWidth < 180;

        final content = Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (loading)
              SizedBox(
                width: compact ? 20 : 28,
                height: compact ? 20 : 28,
                child: const CircularProgressIndicator(
                  strokeWidth: 2.5,
                  color: SteerColors.teal,
                ),
              )
            else
              Icon(
                denied
                    ? Icons.no_photography_outlined
                    : Icons.videocam_off_outlined,
                color: SteerColors.warning,
                size: compact ? 22 : 30,
              ),
            SizedBox(height: compact ? 6 : 10),
            Text(
              compact && !loading ? 'Tap to retry' : title,
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.white,
                fontSize: compact ? 11 : 15,
                fontWeight: FontWeight.w700,
              ),
            ),
            if (!compact && !loading) ...[
              const SizedBox(height: 6),
              Text(
                body,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 12,
                  height: 1.35,
                ),
              ),
              const SizedBox(height: 10),
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: SteerColors.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                ),
                onPressed: onRetry,
                icon: const Icon(Icons.refresh_rounded, size: 18),
                label: const Text('Retry'),
              ),
            ],
          ],
        );

        if (compact) {
          return InkWell(
            onTap: loading ? null : onRetry,
            child: Center(child: content),
          );
        }
        return Align(
          alignment: const Alignment(0, -0.7),
          child: Container(
            constraints: const BoxConstraints(maxWidth: 290),
            margin: const EdgeInsets.symmetric(horizontal: 16),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0xE60B1E26),
              borderRadius: BorderRadius.circular(18),
            ),
            child: content,
          ),
        );
      },
    );
  }
}

/// Robot ESP32-CAM feed (MJPEG). Only shows video while a real ESP32 robot is
/// connected; otherwise (offline / demo) it shows a yellow warning.
class RobotFeedView extends StatelessWidget {
  final AppState appState;
  const RobotFeedView({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final robotOnline = appState.isConnected && !appState.isMock;
        return LayoutBuilder(
          builder: (context, c) {
            final compact = c.maxWidth < 180 || c.maxHeight < 180;
            if (!robotOnline) return _FeedMessage(compact: compact);
            return ValueListenableBuilder<Uint8List?>(
              valueListenable: appState.robotFrame,
              builder: (context, frame, _) {
                if (frame != null) {
                  return Image.memory(
                    frame,
                    fit: BoxFit.cover,
                    gaplessPlayback: true,
                    width: double.infinity,
                    height: double.infinity,
                    errorBuilder: (_, _, _) => _FeedMessage(compact: compact),
                  );
                }
                if (appState.streamError || appState.camStreamUrl.isEmpty) {
                  return _FeedMessage(compact: compact);
                }
                return _FeedMessage(compact: compact, connecting: true);
              },
            );
          },
        );
      },
    );
  }
}

class _FeedMessage extends StatelessWidget {
  final bool compact;
  final bool connecting;
  const _FeedMessage({required this.compact, this.connecting = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF16323E), Color(0xFF0B1E26)],
        ),
      ),
      alignment: Alignment.center,
      padding: const EdgeInsets.all(8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (connecting)
            SizedBox(
              width: compact ? 18 : 26,
              height: compact ? 18 : 26,
              child: const CircularProgressIndicator(
                strokeWidth: 2.5,
                color: SteerColors.teal,
              ),
            )
          else
            Icon(
              Icons.videocam_off_outlined,
              color: SteerColors.warningYellow,
              size: compact ? 22 : 32,
            ),
          SizedBox(height: compact ? 6 : 10),
          Text(
            connecting ? 'Connecting to ESP32-CAM...' : 'No camera detected!',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: connecting ? Colors.white70 : SteerColors.warningYellow,
              fontSize: compact ? 11 : 16,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

// ============================================================================
// SCREEN 1: ONBOARDING
// ============================================================================

class OnboardingScreen extends StatelessWidget {
  final AppState appState;
  const OnboardingScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    final badges = ['1', '2', '3', '4', 'Palm'];

    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SteerBrandHeader(appState: appState),
              const Spacer(flex: 2),
              RichText(
                text: const TextSpan(
                  style: TextStyle(
                    fontSize: 46,
                    fontWeight: FontWeight.w800,
                    color: SteerColors.dark,
                    height: 1.1,
                  ),
                  children: [
                    TextSpan(text: 'Steer with\nyour '),
                    TextSpan(
                      text: 'hand.',
                      style: TextStyle(color: SteerColors.teal),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Drive your robot with finger gestures. No joystick, no buttons.',
                style: TextStyle(
                  color: SteerColors.subtitle,
                  fontSize: 17,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 28),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 14, vertical: 18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: badges.map((b) {
                    return Container(
                      width: 56,
                      height: 62,
                      decoration: BoxDecoration(
                        color: SteerColors.tealLight,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(gestureIconFor(b),
                              size: 18, color: SteerColors.teal),
                          const SizedBox(height: 2),
                          Text(
                            b,
                            style: const TextStyle(
                              color: SteerColors.dark,
                              fontWeight: FontWeight.w800,
                              fontSize: 15,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ),
              const Spacer(flex: 3),
              SizedBox(
                width: double.infinity,
                height: 58,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: SteerColors.teal,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(20),
                    ),
                  ),
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => HomeHubScreen(appState: appState),
                      ),
                    );
                  },
                  child: const Text(
                    'Get started',
                    style: TextStyle(
                      color: SteerColors.dark,
                      fontWeight: FontWeight.w800,
                      fontSize: 18,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// SCREEN 2: HOME HUB
// ============================================================================

class HomeHubScreen extends StatelessWidget {
  final AppState appState;
  const HomeHubScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    void go(Widget w) =>
        Navigator.push(context, MaterialPageRoute(builder: (_) => w));

    final menuItems = [
      ('1', 'Drive', Icons.gamepad_outlined, () {
        appState.setPracticeMode(false);
        go(DriveScreen(appState: appState));
      }),
      ('2', 'Manual control', Icons.sports_esports_outlined,
          () => go(ManualControlScreen(appState: appState))),
      ('3', 'Connect robot', Icons.wifi_tethering_rounded,
          () => go(ConnectRobotScreen(appState: appState))),
      ('4', 'Sensitivity', Icons.tune_rounded,
          () => go(SensitivityScreen(appState: appState))),
      ('5', 'FPV camera', Icons.videocam_outlined,
          () => go(FpvCameraScreen(appState: appState))),
      ('6', 'Gestures', Icons.front_hand_outlined,
          () => go(GesturesScreen(appState: appState))),
      ('7', 'About', Icons.info_outline_rounded,
          () => go(AboutScreen(appState: appState))),
    ];

    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SteerBrandHeader(appState: appState),
                  const SizedBox(height: 22),
                  const Text(
                    'What now?',
                    style: TextStyle(
                      color: SteerColors.dark,
                      fontSize: 32,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                  const SizedBox(height: 18),
                  Expanded(
                    child: GridView.builder(
                      itemCount: menuItems.length,
                      gridDelegate:
                          const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 2,
                        mainAxisSpacing: 14,
                        crossAxisSpacing: 14,
                        childAspectRatio: 1.05,
                      ),
                      itemBuilder: (context, index) {
                        final item = menuItems[index];
                        return InkWell(
                          onTap: item.$4,
                          borderRadius: BorderRadius.circular(24),
                          child: Container(
                            padding: const EdgeInsets.all(18),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(24),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Row(
                                  mainAxisAlignment:
                                      MainAxisAlignment.spaceBetween,
                                  children: [
                                    Container(
                                      width: 44,
                                      height: 44,
                                      decoration: BoxDecoration(
                                        color: SteerColors.tealLight,
                                        borderRadius: BorderRadius.circular(14),
                                      ),
                                      alignment: Alignment.center,
                                      child: Text(
                                        item.$1,
                                        style: const TextStyle(
                                          color: SteerColors.teal,
                                          fontWeight: FontWeight.w800,
                                          fontSize: 17,
                                        ),
                                      ),
                                    ),
                                    Icon(item.$3,
                                        color: SteerColors.subtitle, size: 20),
                                  ],
                                ),
                                Text(
                                  item.$2,
                                  style: const TextStyle(
                                    color: SteerColors.dark,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 18,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                  if (appState.isMoving)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: SizedBox(
                        width: double.infinity,
                        height: 52,
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: SteerColors.danger,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(18),
                            ),
                          ),
                          onPressed: appState.emergencyStop,
                          icon: const Icon(Icons.stop_circle_outlined),
                          label: Text(
                            'EMERGENCY STOP (${appState.commandName} active)',
                            style: const TextStyle(fontWeight: FontWeight.w800),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 3: CONNECT ROBOT
// ============================================================================

class ConnectRobotScreen extends StatefulWidget {
  final AppState appState;
  const ConnectRobotScreen({super.key, required this.appState});

  @override
  State<ConnectRobotScreen> createState() => _ConnectRobotScreenState();
}

class _ConnectRobotScreenState extends State<ConnectRobotScreen> {
  late TextEditingController _ipController;
  late TextEditingController _portController;
  late TextEditingController _camUrlController;

  @override
  void initState() {
    super.initState();
    _ipController = TextEditingController(text: widget.appState.robotIp);
    _portController = TextEditingController(text: widget.appState.udpPort);
    _camUrlController =
        TextEditingController(text: widget.appState.camStreamUrl);
  }

  @override
  void dispose() {
    _ipController.dispose();
    _portController.dispose();
    _camUrlController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.appState;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final okColor = s.isConnected ? SteerColors.teal : const Color(0xFF8B1D1D);
        final statusText = s.isConnected
            ? (s.isMock
                ? 'Demo link active (simulated robot)'
                : 'Connected to ${s.robotIp}:${s.udpPort}  -  ${s.latencyMs} ms')
            : (s.statusMessage ?? 'Robot not found. Check address');

        return Scaffold(
          body: SafeArea(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SteerPageTopBar(
                    appState: s,
                    title: 'Connect robot',
                    trailing: TextButton.icon(
                      onPressed: s.isScanning
                          ? null
                          : () async {
                              await s.scanNetwork();
                              _ipController.text = s.robotIp;
                              _camUrlController.text = s.camStreamUrl;
                            },
                      icon: s.isScanning
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.radar_rounded, size: 18),
                      label: Text(s.isScanning ? 'Scanning...' : 'Auto-Scan'),
                    ),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Robot address on the same network as this phone. Leave empty for camera-only demo.',
                          style: TextStyle(
                            color: SteerColors.subtitle,
                            fontSize: 15,
                            height: 1.4,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            Expanded(
                              flex: 3,
                              child: TextField(
                                controller: _ipController,
                                keyboardType: TextInputType.number,
                                style: const TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.w600,
                                  color: SteerColors.dark,
                                ),
                                decoration: _fieldDecoration(
                                  'ESP32 IP Address',
                                  '192.168.4.1',
                                  highlight: true,
                                ),
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              flex: 1,
                              child: TextField(
                                controller: _portController,
                                keyboardType: TextInputType.number,
                                style: const TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.w600,
                                  color: SteerColors.dark,
                                ),
                                decoration: _fieldDecoration('HTTP Port', '8080'),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        SizedBox(
                          width: double.infinity,
                          height: 54,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: SteerColors.teal,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: s.isTesting
                                ? null
                                : () => s.testConnection(
                                      _ipController.text,
                                      _portController.text,
                                      _camUrlController.text,
                                    ),
                            child: s.isTesting
                                ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2, color: Colors.white),
                                  )
                                : const Text(
                                    'Save and test',
                                    style: TextStyle(
                                      color: SteerColors.dark,
                                      fontWeight: FontWeight.w800,
                                      fontSize: 17,
                                    ),
                                  ),
                          ),
                        ),
                        const SizedBox(height: 14),
                        if (s.hasTestedConnection)
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 16, vertical: 10),
                            decoration: BoxDecoration(
                              color: s.isConnected
                                  ? SteerColors.tealLight
                                  : SteerColors.dangerLight,
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  s.isConnected
                                      ? Icons.check_circle_rounded
                                      : Icons.error_outline_rounded,
                                  size: 18,
                                  color: okColor,
                                ),
                                const SizedBox(width: 8),
                                Flexible(
                                  child: Text(
                                    statusText,
                                    style: TextStyle(
                                      color: s.isConnected
                                          ? SteerColors.dark
                                          : okColor,
                                      fontWeight: FontWeight.w700,
                                      fontSize: 14,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        const SizedBox(height: 8),
                        SwitchListTile(
                          contentPadding: EdgeInsets.zero,
                          activeThumbColor: SteerColors.teal,
                          title: const Text('Demo mode (simulate robot)',
                              style: TextStyle(
                                  fontWeight: FontWeight.w700,
                                  color: SteerColors.dark)),
                          subtitle: Text(kIsWeb
                              ? 'Always on in the browser (cannot reach the robot)'
                              : 'No commands are sent over Wi-Fi'),
                          value: s.isMock,
                          onChanged: kIsWeb ? null : s.setMockMode,
                        ),
                        SwitchListTile(
                          contentPadding: EdgeInsets.zero,
                          activeThumbColor: SteerColors.teal,
                          title: const Text('Send even without status reply',
                              style: TextStyle(
                                  fontWeight: FontWeight.w700,
                                  color: SteerColors.dark)),
                          subtitle: const Text(
                              'Send commands even if /status gets no reply'),
                          value: s.oneWayMode,
                          onChanged: s.isMock ? null : s.setOneWayMode,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Optional: add the ESP32-CAM address to turn on live FPV video.',
                          style: TextStyle(
                            color: SteerColors.subtitle,
                            fontSize: 15,
                            height: 1.4,
                          ),
                        ),
                        const SizedBox(height: 14),
                        TextField(
                          controller: _camUrlController,
                          style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                            color: SteerColors.dark,
                          ),
                          decoration: InputDecoration(
                            labelText: 'ESP32-CAM MJPEG Stream URL',
                            hintText: 'http://192.168.4.2:81/stream',
                            prefixIcon: const Icon(Icons.videocam_outlined),
                            contentPadding: const EdgeInsets.symmetric(
                                horizontal: 14, vertical: 14),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  InputDecoration _fieldDecoration(String label, String hint,
      {bool highlight = false}) {
    return InputDecoration(
      labelText: label,
      hintText: hint,
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 16),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: BorderSide(
          color: highlight ? SteerColors.teal : Colors.black26,
          width: highlight ? 1.5 : 1,
        ),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(16),
        borderSide: const BorderSide(color: SteerColors.teal, width: 2),
      ),
    );
  }
}

// ============================================================================
// SCREEN 4: DRIVE
// ============================================================================

class DriveScreen extends StatefulWidget {
  final AppState appState;
  const DriveScreen({super.key, required this.appState});

  @override
  State<DriveScreen> createState() => _DriveScreenState();
}

class _DriveScreenState extends State<DriveScreen> {
  AppState get appState => widget.appState;

  @override
  void initState() {
    super.initState();
    appState.acquireCamera();
    appState.acquireTracking(); // live hand detection (Drive AND Practice)
  }

  @override
  void dispose() {
    appState.sendStopBurst(); // never leave the robot moving
    appState.releaseTracking();
    appState.releaseCamera();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    const controls = [
      ('1', 'FWD'),
      ('2', 'BACK'),
      ('3', 'LEFT'),
      ('4', 'RIGHT'),
      ('Palm', 'STOP'),
    ];

    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final isHandLost = appState.activeGesture == 'LOST';
        final hasRealLandmarks =
            appState.handLandmarks != null || appState.handObs != null;

        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(
                    appState: appState,
                    title: appState.isPracticeMode ? 'Practice Drive' : 'Drive',
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          tooltip: 'Toggle Hand Skeleton Overlay',
                          onPressed: appState.toggleLandmarks,
                          icon: Icon(
                            Icons.hub_outlined,
                            color: appState.showLandmarks
                                ? SteerColors.teal
                                : SteerColors.subtitle,
                          ),
                        ),
                        IconButton(
                          tooltip: 'Mirror Camera Feed',
                          onPressed: appState.toggleMirror,
                          icon: Icon(
                            Icons.flip_rounded,
                            color: appState.mirrorCamera
                                ? SteerColors.teal
                                : SteerColors.subtitle,
                          ),
                        ),
                        IconButton(
                          tooltip: 'Switch Front/Rear Camera',
                          onPressed: appState.toggleFrontCamera,
                          icon: const Icon(Icons.cameraswitch_outlined,
                              color: SteerColors.dark),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(28),
                      child: Container(
                        width: double.infinity,
                        color: SteerColors.cameraBg,
                        child: Stack(
                          children: [
                            Positioned.fill(
                              child: HandCameraView(
                                appState: appState,
                                simulateIfIdle: true,
                              ),
                            ),
                            Positioned(
                              top: 16,
                              left: 16,
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 14, vertical: 7),
                                decoration: BoxDecoration(
                                  color: isHandLost
                                      ? SteerColors.warning
                                      : SteerColors.teal,
                                  borderRadius: BorderRadius.circular(20),
                                ),
                                child: Text(
                                  isHandLost
                                      ? 'Hand Lost - Auto-Brake'
                                      : appState.isPracticeMode
                                          ? 'Practice mode (Motors Off)'
                                          : 'Gesture mode',
                                  style: const TextStyle(
                                    color: SteerColors.dark,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 13,
                                  ),
                                ),
                              ),
                            ),
                            Positioned(
                              top: 18,
                              right: 16,
                              child: Text(
                                '${appState.isFrontCamera ? "Front Cam" : "Rear Cam"}${appState.mirrorCamera ? " (Mirrored)" : ""}',
                                style: const TextStyle(
                                  color: Colors.white70,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                            Center(
                              child: CustomPaint(
                                painter: DashedFramePainter(),
                                child: SizedBox(
                                  width: 210,
                                  height: 250,
                                  child: Center(
                                    child: Text(
                                      isHandLost
                                          ? 'No hand in frame\nMotors Locked'
                                          : (hasRealLandmarks ||
                                                  appState.cameraStatus !=
                                                      CameraStatus.ready
                                              ? ''
                                              : 'Show your hand'),
                                      textAlign: TextAlign.center,
                                      style: TextStyle(
                                        color: isHandLost
                                            ? SteerColors.warning
                                            : Colors.white70,
                                        fontSize: 16,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ),
                            Positioned(
                              bottom: 12,
                              right: 14,
                              child: TextButton.icon(
                                style: TextButton.styleFrom(
                                  backgroundColor: Colors.black38,
                                  foregroundColor: Colors.white70,
                                ),
                                onPressed: () => appState
                                    .setGesture(isHandLost ? 'Palm' : 'LOST'),
                                icon: Icon(
                                  isHandLost
                                      ? Icons.visibility_rounded
                                      : Icons.visibility_off_outlined,
                                  size: 16,
                                ),
                                label: Text(
                                  isHandLost
                                      ? 'Restore Hand'
                                      : 'Simulate Hand Lost',
                                  style: const TextStyle(fontSize: 12),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(24),
                    ),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Flexible(
                              child: Text(
                                appState.commandName,
                                overflow: TextOverflow.ellipsis,
                                style: TextStyle(
                                  color: isHandLost
                                      ? SteerColors.danger
                                      : SteerColors.dark,
                                  fontSize: 22,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                            Text(
                              'Link ${appState.linkQuality}%  Battery ${appState.batteryLevel}%',
                              style: const TextStyle(
                                color: SteerColors.subtitle,
                                fontSize: 13,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Align(
                          alignment: Alignment.centerLeft,
                          child: ValueListenableBuilder<String>(
                            valueListenable: appState.packetNotifier,
                            builder: (_, p, _) => Text(
                              appState.isPracticeMode
                                  ? 'Practice: motors off, nothing is sent'
                                  : (appState.isConnected
                                      ? 'TX ${appState.isMock ? "(demo) " : ""}$p'
                                      : 'Not connected - commands are not sent'),
                              style: const TextStyle(
                                color: SteerColors.subtitle,
                                fontSize: 12,
                                fontFamily: 'monospace',
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Align(
                          alignment: Alignment.centerLeft,
                          child: Text(
                            appState.trackingLabel,
                            style: TextStyle(
                              color: (appState.handObs != null ||
                                      appState.handLandmarks != null)
                                  ? SteerColors.teal
                                  : SteerColors.subtitle,
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        const SizedBox(height: 10),
                        SteerMeter(appState: appState),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: controls.map((item) {
                            final isSelected =
                                appState.activeGesture == item.$1 &&
                                    !appState.estop ||
                                (appState.estop && item.$1 == 'Palm');
                            return GestureDetector(
                              onTap: () => appState.setGesture(item.$1),
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 160),
                                width: 58,
                                height: 68,
                                decoration: BoxDecoration(
                                  color: isSelected
                                      ? SteerColors.teal
                                      : const Color(0xFFF1F5F5),
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Text(
                                      item.$1,
                                      style: const TextStyle(
                                        color: SteerColors.dark,
                                        fontWeight: FontWeight.w800,
                                        fontSize: 17,
                                      ),
                                    ),
                                    const SizedBox(height: 4),
                                    Text(
                                      item.$2,
                                      style: const TextStyle(
                                        color: SteerColors.dark,
                                        fontWeight: FontWeight.w700,
                                        fontSize: 11,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          }).toList(),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: SteerColors.dark,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (_) =>
                                      FpvCameraScreen(appState: appState),
                                ),
                              );
                            },
                            child: const Text(
                              'FPV',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: SteerColors.danger,
                              foregroundColor: Colors.white,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: appState.emergencyStop,
                            child: const Text(
                              'STOP',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 5: FPV CAMERA
// ============================================================================

class FpvCameraScreen extends StatefulWidget {
  final AppState appState;
  const FpvCameraScreen({super.key, required this.appState});

  @override
  State<FpvCameraScreen> createState() => _FpvCameraScreenState();
}

class _FpvCameraScreenState extends State<FpvCameraScreen> {
  AppState get appState => widget.appState;

  @override
  void initState() {
    super.initState();
    appState.acquireCamera();
    appState.acquireStream();
  }

  @override
  void dispose() {
    appState.releaseCamera();
    appState.releaseStream();
    super.dispose();
  }

  Widget _box(Widget child, String badge) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Container(
        width: double.infinity,
        color: SteerColors.cameraBg,
        child: Stack(
          fit: StackFit.expand,
          children: [
            child,
            Positioned(
              top: 12,
              left: 12,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  badge,
                  style: const TextStyle(
                    color: SteerColors.dark,
                    fontWeight: FontWeight.w800,
                    fontSize: 12,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final robot = RobotFeedView(appState: appState);
        final hand = HandCameraView(
            appState: appState, showOverlay: appState.showLandmarks);
        final main = appState.isPipSwapped ? hand : robot;
        final pip = appState.isPipSwapped ? robot : hand;
        final robotBadge = appState.streamLive ? 'Live' : 'Robot (offline)';

        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(
                    appState: appState,
                    title: 'FPV camera',
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          tooltip: 'Toggle Horizontal Split / PiP View',
                          onPressed: appState.toggleSplitView,
                          icon: Icon(
                            appState.isHorizontalSplit
                                ? Icons.picture_in_picture_alt_rounded
                                : Icons.vertical_split_rounded,
                            color: SteerColors.teal,
                          ),
                        ),
                        IconButton(
                          tooltip: 'Switch Front/Rear Hand Cam',
                          onPressed: appState.toggleFrontCamera,
                          icon: const Icon(Icons.cameraswitch_outlined,
                              color: SteerColors.dark),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  Expanded(
                    child: appState.isHorizontalSplit
                        ? Column(
                            children: [
                              Expanded(child: _box(robot, robotBadge)),
                              const SizedBox(height: 10),
                              Expanded(
                                child: _box(
                                  hand,
                                  appState.isFrontCamera ? 'Front' : 'Rear',
                                ),
                              ),
                            ],
                          )
                        : Stack(
                            children: [
                              Positioned.fill(
                                child: _box(
                                  main,
                                  appState.isPipSwapped ? 'Hand' : robotBadge,
                                ),
                              ),
                              Positioned(
                                bottom: 18,
                                right: 18,
                                child: GestureDetector(
                                  onTap: appState.togglePipSwap,
                                  child: Container(
                                    width: 115,
                                    height: 145,
                                    decoration: BoxDecoration(
                                      color: const Color(0xFF0B1E26),
                                      borderRadius: BorderRadius.circular(20),
                                      border: Border.all(
                                          color: SteerColors.teal, width: 2),
                                    ),
                                    child: ClipRRect(
                                      borderRadius: BorderRadius.circular(18),
                                      child: Stack(
                                        fit: StackFit.expand,
                                        children: [
                                          pip,
                                          const Positioned(
                                            top: 6,
                                            right: 6,
                                            child: Icon(
                                              Icons.swap_horiz_rounded,
                                              color: SteerColors.teal,
                                              size: 22,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                  ),
                  const SizedBox(height: 14),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(22),
                    ),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('Command',
                                style: TextStyle(
                                    color: SteerColors.subtitle, fontSize: 13)),
                            const SizedBox(height: 2),
                            Text(
                              appState.commandName,
                              style: const TextStyle(
                                color: SteerColors.teal,
                                fontSize: 22,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            const Text('Latency',
                                style: TextStyle(
                                    color: SteerColors.subtitle, fontSize: 13)),
                            const SizedBox(height: 2),
                            Text(
                              appState.isConnected
                                  ? '${appState.latencyMs} ms'
                                  : '-- ms',
                              style: const TextStyle(
                                color: SteerColors.dark,
                                fontSize: 22,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: SteerColors.danger,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      onPressed: appState.emergencyStop,
                      child: const Text(
                        'STOP',
                        style:
                            TextStyle(fontWeight: FontWeight.w800, fontSize: 17),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 6: GESTURES GUIDE
// ============================================================================

class GesturesScreen extends StatelessWidget {
  final AppState appState;
  const GesturesScreen({super.key, required this.appState});

  static const _tiltGuide = <(String, IconData, String, String)>[
    ('1', Icons.touch_app_rounded, 'Forward', 'One finger up'),
    ('2', Icons.pan_tool_alt_rounded, 'Backward', 'Two fingers up'),
    ('Palm', Icons.pan_tool_rounded, 'Stop',
        'Closed fist or open palm: stops the car at once'),
    ('TILT', Icons.screen_rotation_alt_rounded, 'Steer',
        'Tilt your hand left or right like a steering wheel. Works together with forward and backward. The more you tilt, the harder it turns.'),
  ];

  static const _classicGuide = <(String, IconData, String, String)>[
    ('1', Icons.touch_app_rounded, 'Forward', 'One finger up'),
    ('2', Icons.pan_tool_alt_rounded, 'Backward', 'Two fingers up'),
    ('3', Icons.back_hand_outlined, 'Turn left', 'Three fingers up'),
    ('4', Icons.front_hand_outlined, 'Turn right', 'Four fingers up'),
    ('Palm', Icons.pan_tool_rounded, 'Stop',
        'Open palm or fist, stops at once'),
  ];

  Widget _modeChip(String title, String sub, bool selected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 10),
        decoration: BoxDecoration(
          color: selected ? SteerColors.teal : const Color(0xFFF1F5F5),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          children: [
            Text(
              title,
              style: TextStyle(
                color: selected ? Colors.white : SteerColors.dark,
                fontWeight: FontWeight.w800,
                fontSize: 15,
              ),
            ),
            const SizedBox(height: 2),
            Text(
              sub,
              style: TextStyle(
                color: selected ? Colors.white70 : SteerColors.subtitle,
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final tilt = appState.controlMode == ControlMode.tilt;
        final guide = tilt ? _tiltGuide : _classicGuide;
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(appState: appState, title: 'Gestures'),
                  const SizedBox(height: 14),
                  Container(
                    padding: const EdgeInsets.all(6),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Row(
                      children: [
                        Expanded(
                          child: _modeChip(
                            'Tilt steering',
                            'Recommended',
                            tilt,
                            () => appState.setControlMode(ControlMode.tilt),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: _modeChip(
                            'Classic',
                            '1-4 fingers',
                            !tilt,
                            () => appState.setControlMode(ControlMode.classic),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: ListView.separated(
                      itemCount: guide.length + 1,
                      separatorBuilder: (_, _) => const SizedBox(height: 12),
                      itemBuilder: (context, index) {
                        if (index == guide.length) {
                          return Container(
                            padding: const EdgeInsets.all(14),
                            decoration: BoxDecoration(
                              color: SteerColors.tealLight,
                              borderRadius: BorderRadius.circular(18),
                            ),
                            child: Text(
                              tilt
                                  ? 'Hand out of view for half a second: the car brakes and the steering goes straight. Tilt steering needs the Android app with hand tracking; if the skeleton never appears, use Classic or Manual control.'
                                  : 'Classic mode does one thing at a time, so the car cannot turn while driving. Use Tilt steering for that.',
                              style: const TextStyle(
                                color: SteerColors.dark,
                                fontSize: 13,
                                height: 1.4,
                              ),
                            ),
                          );
                        }
                        final g = guide[index];
                        return Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(22),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                width: 56,
                                height: 56,
                                decoration: BoxDecoration(
                                  color: SteerColors.tealLight,
                                  borderRadius: BorderRadius.circular(16),
                                ),
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(g.$2, size: 18, color: SteerColors.teal),
                                    Text(
                                      g.$1,
                                      style: const TextStyle(
                                        color: SteerColors.teal,
                                        fontWeight: FontWeight.w800,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(width: 16),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      g.$3,
                                      style: const TextStyle(
                                        color: SteerColors.dark,
                                        fontWeight: FontWeight.w800,
                                        fontSize: 18,
                                      ),
                                    ),
                                    const SizedBox(height: 3),
                                    Text(
                                      g.$4,
                                      style: const TextStyle(
                                        color: SteerColors.subtitle,
                                        fontSize: 14,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    height: 56,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: SteerColors.teal,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      onPressed: () {
                        appState.setPracticeMode(true);
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (_) => DriveScreen(appState: appState),
                          ),
                        );
                      },
                      child: const Text(
                        'Practice mode (motors off)',
                        style: TextStyle(
                          color: SteerColors.dark,
                          fontWeight: FontWeight.w800,
                          fontSize: 17,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 7: SENSITIVITY & CALIBRATION
// ============================================================================

class SensitivityScreen extends StatelessWidget {
  final AppState appState;
  const SensitivityScreen({super.key, required this.appState});

  Widget _section(String title, String sub) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 8, 4, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              color: SteerColors.dark,
              fontWeight: FontWeight.w800,
              fontSize: 20,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            sub,
            style: const TextStyle(color: SteerColors.subtitle, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _slider({
    required String title,
    required String valueLabel,
    required String subtitle,
    required double value,
    required double min,
    required double max,
    required ValueChanged<double> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Container(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 14),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(24),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Expanded(
                  child: Text(
                    title,
                    style: const TextStyle(
                      color: SteerColors.dark,
                      fontWeight: FontWeight.w800,
                      fontSize: 16,
                    ),
                  ),
                ),
                Text(
                  valueLabel,
                  style: const TextStyle(
                    color: SteerColors.teal,
                    fontWeight: FontWeight.w800,
                    fontSize: 16,
                  ),
                ),
              ],
            ),
            SliderTheme(
              data: SliderThemeData(
                activeTrackColor: SteerColors.teal,
                inactiveTrackColor: const Color(0xFFE4ECEB),
                thumbColor: SteerColors.teal,
                trackHeight: 8,
                overlayColor: SteerColors.teal.withValues(alpha: 0.15),
              ),
              child: Slider(
                value: value.clamp(min, max).toDouble(),
                min: min,
                max: max,
                onChanged: onChanged,
                onChangeEnd: (_) => appState.saveSettings(),
              ),
            ),
            Text(
              subtitle,
              style: const TextStyle(color: SteerColors.subtitle, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }

  static String _pct(double v) => '${(v * 100).round()}%';

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        final a = appState;
        final trim = a.steeringTrim.round();
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(appState: appState, title: 'Sensitivity'),
                  const SizedBox(height: 14),
                  Expanded(
                    child: ListView(
                      children: [
                        _section('Steering',
                            'Settings for the steering motor and for turning with your hand.'),
                        _slider(
                          title: 'Turn strength',
                          valueLabel: _pct(a.turnStrength),
                          subtitle:
                              'Steering power at a full turn. Lower = gentler, wider turns.',
                          value: a.turnStrength,
                          min: 0.2,
                          max: 1.0,
                          onChanged: (v) => a.updateSensitivity(turn: v),
                        ),
                        _slider(
                          title: 'Turn minimum power',
                          valueLabel: _pct(a.turnMin),
                          subtitle:
                              'Smallest power that makes the wheels move. Raise it if small tilts do nothing.',
                          value: a.turnMin,
                          min: 0.1,
                          max: 0.6,
                          onChanged: (v) => a.updateSensitivity(turnMinimum: v),
                        ),
                        _slider(
                          title: 'Turn response',
                          valueLabel: a.turnResponse < 0.34
                              ? 'Smooth'
                              : (a.turnResponse < 0.67 ? 'Medium' : 'Quick'),
                          subtitle:
                              'How fast the steering follows your hand. Smooth = gradual, Quick = instant.',
                          value: a.turnResponse,
                          min: 0.0,
                          max: 1.0,
                          onChanged: (v) => a.updateSensitivity(response: v),
                        ),
                        _slider(
                          title: 'Tilt for full turn',
                          valueLabel: '${a.tiltFullDeg.round()}°',
                          subtitle:
                              'How far you tilt your hand for a full turn. Lower = more sensitive.',
                          value: a.tiltFullDeg,
                          min: 20,
                          max: 60,
                          onChanged: (v) =>
                              a.updateSensitivity(tiltFull: v.roundToDouble()),
                        ),
                        _slider(
                          title: 'Tilt dead zone',
                          valueLabel: '${a.tiltDeadDeg.round()}°',
                          subtitle:
                              'Tilts smaller than this are ignored, so the car drives straight.',
                          value: a.tiltDeadDeg,
                          min: 5,
                          max: 25,
                          onChanged: (v) =>
                              a.updateSensitivity(tiltDead: v.roundToDouble()),
                        ),
                        _slider(
                          title: 'Left / right balance',
                          valueLabel: trim == 0
                              ? 'Centre'
                              : (trim > 0 ? '+$trim R' : '$trim L'),
                          subtitle:
                              'Give one side more steering power if the car turns more one way.',
                          value: a.steeringTrim,
                          min: -10,
                          max: 10,
                          onChanged: (v) =>
                              a.updateSensitivity(trim: v.roundToDouble()),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 20, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(24),
                            ),
                            child: SwitchListTile(
                              contentPadding: EdgeInsets.zero,
                              activeThumbColor: SteerColors.teal,
                              title: const Text(
                                'Invert steering',
                                style: TextStyle(
                                  fontWeight: FontWeight.w800,
                                  color: SteerColors.dark,
                                ),
                              ),
                              subtitle: const Text(
                                'Turn this on if tilting your hand right steers the car left.',
                              ),
                              value: a.invertSteer,
                              onChanged: a.setInvertSteer,
                            ),
                          ),
                        ),
                        Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: Container(
                            padding: const EdgeInsets.all(20),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(24),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Test steering',
                                  style: TextStyle(
                                    color: SteerColors.dark,
                                    fontWeight: FontWeight.w800,
                                    fontSize: 16,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  a.isConnected
                                      ? 'Sends a short full-turn pulse so you can tune the settings above. The drive motor does not run.'
                                      : 'Connect the robot first (Connect robot).',
                                  style: const TextStyle(
                                    color: SteerColors.subtitle,
                                    fontSize: 13,
                                  ),
                                ),
                                const SizedBox(height: 12),
                                Row(
                                  children: [
                                    Expanded(
                                      child: OutlinedButton.icon(
                                        onPressed: a.isConnected
                                            ? () => a.testSteer(-1)
                                            : null,
                                        icon: const Icon(Icons.west_rounded),
                                        label: const Text('Test left'),
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    Expanded(
                                      child: OutlinedButton.icon(
                                        onPressed: a.isConnected
                                            ? () => a.testSteer(1)
                                            : null,
                                        icon: const Icon(Icons.east_rounded),
                                        label: const Text('Test right'),
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 6),
                        _section('Driving', 'Forward and backward power.'),
                        _slider(
                          title: 'Drive speed',
                          valueLabel: _pct(a.maxMotorSpeed),
                          subtitle: 'Top speed the car will drive at.',
                          value: a.maxMotorSpeed,
                          min: 0.2,
                          max: 1.0,
                          onChanged: (v) => a.updateSensitivity(speed: v),
                        ),
                        _slider(
                          title: 'Minimum drive power',
                          valueLabel: _pct(a.minPwmDeadband),
                          subtitle:
                              'Lowest drive power that gets the gearbox moving.',
                          value: a.minPwmDeadband,
                          min: 0.15,
                          max: 0.50,
                          onChanged: (v) => a.updateSensitivity(deadband: v),
                        ),
                        const SizedBox(height: 6),
                        _section('Hand detection', 'How readily gestures are recognised.'),
                        _slider(
                          title: 'Gesture detection',
                          valueLabel: _pct(a.gestureSensitivity),
                          subtitle:
                              'Higher reacts faster but may misread. Stopping always reacts at once.',
                          value: a.gestureSensitivity,
                          min: 0.1,
                          max: 1.0,
                          onChanged: (v) => a.updateSensitivity(gesture: v),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.white,
                              foregroundColor: SteerColors.dark,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: () {
                              a.resetSensitivityDefaults();
                              a.saveSettings();
                            },
                            child: const Text(
                              'Default',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 14),
                      Expanded(
                        child: SizedBox(
                          height: 56,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: SteerColors.teal,
                              foregroundColor: SteerColors.dark,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(18),
                              ),
                            ),
                            onPressed: () {
                              a.saveSettings();
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text('Settings saved.'),
                                  duration: Duration(seconds: 1),
                                ),
                              );
                              Navigator.pop(context);
                            },
                            child: const Text(
                              'Save',
                              style: TextStyle(
                                  fontWeight: FontWeight.w800, fontSize: 17),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

// ============================================================================
// SCREEN 8: ABOUT
// ============================================================================

class AboutScreen extends StatelessWidget {
  final AppState appState;
  const AboutScreen({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SteerPageTopBar(appState: appState, title: 'About S.T.E.E.R.'),
              const SizedBox(height: 18),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(24),
                ),
                child: const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Smart Teleoperated ESP32 Educational Robot',
                      style: TextStyle(
                        color: SteerColors.dark,
                        fontWeight: FontWeight.w800,
                        fontSize: 18,
                      ),
                    ),
                    SizedBox(height: 10),
                    Text(
                      '- Vision Pipeline: 21-point MediaPipe Hand Landmarks\n'
                      '- Transport: HTTP GET /move over local Wi-Fi (on change + every 300 ms)\n'
                      '- Request: /move?drive=1&steer=-1&speed=55&turn=70\n'
                      '- Heartbeat: app polls /status every second\n'
                      '- Steering: tilt your hand to steer while driving, or use Manual control\n'
                      '- Failsafe: zero-PWM brake when hand leaves frame\n'
                      '- Hardware: ESP32 + L298N / TB6612FNG 2WD chassis + optional ESP32-CAM',
                      style: TextStyle(
                        color: SteerColors.subtitle,
                        fontSize: 15,
                        height: 1.6,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ============================================================================
// PAINTERS
// ============================================================================

/// Dashed guide frame: keep your hand inside it.
class DashedFramePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final borderPaint = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final rrect = RRect.fromRectAndRadius(
      Offset.zero & size,
      const Radius.circular(28),
    );
    final path = Path()..addRRect(rrect);
    for (final metric in path.computeMetrics()) {
      double distance = 0;
      while (distance < metric.length) {
        final next = math.min(distance + 8, metric.length);
        canvas.drawPath(metric.extractPath(distance, next), borderPaint);
        distance += 14;
      }
    }
  }

  @override
  bool shouldRepaint(covariant DashedFramePainter old) => false;
}

/// Paints a 21-point hand skeleton. With [center] it is anchored on the
/// detected hand (mapped through the same "cover" crop as the camera preview);
/// without it, it is drawn at the middle of the viewport as a demo.
class HandOverlayPainter extends CustomPainter {
  final List<Offset> template; // palm-radius units
  final Offset? center; // normalized upright-image coordinates
  final double radiusNormH;
  final double aspect; // upright image width / height
  final bool mirror;

  HandOverlayPainter({
    required this.template,
    this.center,
    this.radiusNormH = 0.1,
    this.aspect = 0.75,
    required this.mirror,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (template.length != 21) return;
    final c = center;
    late Offset origin;
    late double unit;
    if (c == null) {
      origin = Offset(size.width * 0.5, size.height * 0.55);
      unit = math.min(size.width, size.height) * 0.16;
    } else {
      final scale = math.max(size.width / aspect, size.height);
      final drawnW = aspect * scale;
      final drawnH = scale;
      final ox = (size.width - drawnW) / 2;
      final oy = (size.height - drawnH) / 2;
      final nx = mirror ? 1 - c.dx : c.dx;
      origin = Offset(ox + nx * drawnW, oy + c.dy * drawnH);
      unit = radiusNormH * drawnH;
    }
    unit = math.max(8.0, math.min(140.0, unit));

    Offset p(Offset t) =>
        origin + Offset((mirror ? -t.dx : t.dx) * unit, t.dy * unit);

    final line = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.85)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    final dot = Paint()..color = Colors.white;
    for (final e in kHandConnections) {
      canvas.drawLine(p(template[e[0]]), p(template[e[1]]), line);
    }
    for (final t in template) {
      canvas.drawCircle(p(t), 3.4, dot);
    }
  }

  @override
  bool shouldRepaint(covariant HandOverlayPainter old) => true;
}

/// Draws real 21-point landmarks (normalized) over the whole camera viewport.
class LandmarkOverlayPainter extends CustomPainter {
  final List<Offset> landmarks;
  final bool flip;
  LandmarkOverlayPainter(this.landmarks, this.flip);

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.length != 21) return;
    Offset p(Offset n) =>
        Offset((flip ? 1 - n.dx : n.dx) * size.width, n.dy * size.height);
    final line = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.8)
      ..strokeWidth = 2.5;
    final dot = Paint()..color = Colors.white;
    for (final c in kHandConnections) {
      canvas.drawLine(p(landmarks[c[0]]), p(landmarks[c[1]]), line);
    }
    for (final l in landmarks) {
      canvas.drawCircle(p(l), 3.5, dot);
    }
  }

  @override
  bool shouldRepaint(covariant LandmarkOverlayPainter old) => true;
}

/// Draws the real 21-point skeleton. Landmarks are upright + unmirrored and are
/// mapped through the same "cover" crop as the camera preview, so the skeleton
/// sits exactly on the hand.
class HandLandmarksPainter extends CustomPainter {
  final List<Offset> landmarks;
  final double aspect; // upright image width / height
  final bool mirror;
  HandLandmarksPainter({
    required this.landmarks,
    required this.aspect,
    required this.mirror,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.length != 21 || aspect <= 0) return;
    final scale = math.max(size.width / aspect, size.height);
    final drawnW = aspect * scale;
    final drawnH = scale;
    final ox = (size.width - drawnW) / 2;
    final oy = (size.height - drawnH) / 2;
    Offset p(Offset n) => Offset(
          ox + (mirror ? 1 - n.dx : n.dx) * drawnW,
          oy + n.dy * drawnH,
        );
    final line = Paint()
      ..color = SteerColors.teal.withValues(alpha: 0.9)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    final dot = Paint()..color = Colors.white;
    final tip = Paint()..color = SteerColors.warningYellow;
    for (final e in kHandConnections) {
      canvas.drawLine(p(landmarks[e[0]]), p(landmarks[e[1]]), line);
    }
    const tips = {4, 8, 12, 16, 20};
    for (var i = 0; i < 21; i++) {
      canvas.drawCircle(p(landmarks[i]), tips.contains(i) ? 5 : 3.5,
          tips.contains(i) ? tip : dot);
    }
  }

  @override
  bool shouldRepaint(covariant HandLandmarksPainter old) => true;
}


// ============================================================================
// STEERING METER + MANUAL CONTROL
// ============================================================================

/// Live steering bar: hand tilt and how much the car is steering.
class SteerMeter extends StatelessWidget {
  final AppState appState;
  const SteerMeter({super.key, required this.appState});

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<(int, int, int, int, int)>(
      valueListenable: appState.ctlNotifier,
      builder: (context, v, _) {
        final steerPct = v.$2;
        final tilt = v.$3;
        final tiltMode = appState.controlMode == ControlMode.tilt;
        final label = steerPct.abs() < 5
            ? 'straight'
            : (steerPct < 0 ? 'left ${-steerPct}%' : 'right $steerPct%');
        final text = tiltMode
            ? 'Tilt $tilt°  ->  steering $label'
            : 'Steering $label';
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              text,
              style: const TextStyle(
                color: SteerColors.subtitle,
                fontSize: 13,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(height: 6),
            LayoutBuilder(
              builder: (context, box) {
                final w = box.maxWidth;
                final half = w / 2;
                final len = half * (steerPct.abs() / 100.0).clamp(0.0, 1.0);
                return SizedBox(
                  height: 10,
                  child: Stack(
                    children: [
                      Container(
                        decoration: BoxDecoration(
                          color: const Color(0xFFE4ECEB),
                          borderRadius: BorderRadius.circular(5),
                        ),
                      ),
                      Positioned(
                        left: steerPct < 0 ? half - len : half,
                        width: len,
                        top: 0,
                        bottom: 0,
                        child: Container(
                          decoration: BoxDecoration(
                            color: SteerColors.teal,
                            borderRadius: BorderRadius.circular(5),
                          ),
                        ),
                      ),
                      Positioned(
                        left: half - 1,
                        width: 2,
                        top: 0,
                        bottom: 0,
                        child: Container(color: SteerColors.dark),
                      ),
                    ],
                  ),
                );
              },
            ),
          ],
        );
      },
    );
  }
}

/// A button that is "on" only while a finger is held on it.
class _HoldButton extends StatefulWidget {
  final String label;
  final IconData icon;
  final void Function(bool down) onHold;
  const _HoldButton({
    required this.label,
    required this.icon,
    required this.onHold,
  });

  @override
  State<_HoldButton> createState() => _HoldButtonState();
}

class _HoldButtonState extends State<_HoldButton> {
  bool _down = false;

  void _set(bool v) {
    if (_down == v) return;
    setState(() => _down = v);
    widget.onHold(v);
  }

  @override
  void dispose() {
    if (_down) widget.onHold(false);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Listener(
      behavior: HitTestBehavior.opaque,
      onPointerDown: (_) => _set(true),
      onPointerUp: (_) => _set(false),
      onPointerCancel: (_) => _set(false),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 80),
        decoration: BoxDecoration(
          color: _down ? SteerColors.teal : Colors.white,
          borderRadius: BorderRadius.circular(22),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              widget.icon,
              size: 34,
              color: _down ? Colors.white : SteerColors.dark,
            ),
            const SizedBox(height: 2),
            Text(
              widget.label,
              style: TextStyle(
                color: _down ? Colors.white : SteerColors.dark,
                fontWeight: FontWeight.w800,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Horizontal steering control: drag the knob left or right; it springs back
/// to the centre when released. Proportional, so a small drag = a gentle turn.
class _SteerPad extends StatefulWidget {
  final void Function(double v) onSteer;
  const _SteerPad({required this.onSteer});

  @override
  State<_SteerPad> createState() => _SteerPadState();
}

class _SteerPadState extends State<_SteerPad> {
  double _v = 0; // -1 (left) .. 1 (right)
  static const double _margin = 44;

  void _update(Offset local, double width) {
    final usable = width / 2 - _margin;
    if (usable <= 0) return;
    var v = ((local.dx - width / 2) / usable).clamp(-1.0, 1.0).toDouble();
    if (v.abs() < 0.06) v = 0;
    setState(() => _v = v);
    widget.onSteer(v);
  }

  void _release() {
    if (_v != 0) setState(() => _v = 0);
    widget.onSteer(0);
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, box) {
        final w = box.maxWidth;
        final usable = w / 2 - _margin;
        final knobX = w / 2 + _v * (usable > 0 ? usable : 0);
        final active = _v != 0;
        return Listener(
          behavior: HitTestBehavior.opaque,
          onPointerDown: (e) => _update(e.localPosition, w),
          onPointerMove: (e) => _update(e.localPosition, w),
          onPointerUp: (_) => _release(),
          onPointerCancel: (_) => _release(),
          child: Container(
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(28),
            ),
            child: Stack(
              children: [
                Positioned(
                  left: w / 2 - 1,
                  top: 28,
                  bottom: 28,
                  width: 2,
                  child: Container(color: const Color(0xFFE4ECEB)),
                ),
                const Positioned(
                  left: 18,
                  top: 12,
                  child: Text(
                    'LEFT',
                    style: TextStyle(
                      color: SteerColors.subtitle,
                      fontWeight: FontWeight.w800,
                      fontSize: 12,
                    ),
                  ),
                ),
                const Positioned(
                  right: 18,
                  top: 12,
                  child: Text(
                    'RIGHT',
                    style: TextStyle(
                      color: SteerColors.subtitle,
                      fontWeight: FontWeight.w800,
                      fontSize: 12,
                    ),
                  ),
                ),
                Positioned(
                  left: knobX - 36,
                  top: 0,
                  bottom: 0,
                  width: 72,
                  child: Center(
                    child: Container(
                      width: 72,
                      height: 72,
                      decoration: BoxDecoration(
                        color: active
                            ? SteerColors.teal
                            : const Color(0xFFF1F5F5),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        Icons.swap_horiz_rounded,
                        size: 32,
                        color: active ? Colors.white : SteerColors.dark,
                      ),
                    ),
                  ),
                ),
                Positioned(
                  left: 0,
                  right: 0,
                  bottom: 10,
                  child: Center(
                    child: Text(
                      active ? '${(_v.abs() * 100).round()}%' : 'Drag to steer',
                      style: const TextStyle(
                        color: SteerColors.subtitle,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

Widget _miniSlider({
  required String label,
  required double value,
  required double min,
  required double max,
  required ValueChanged<double> onChanged,
  required VoidCallback onEnd,
}) {
  return Row(
    children: [
      SizedBox(
        width: 96,
        child: Text(
          label,
          style: const TextStyle(
            color: SteerColors.dark,
            fontWeight: FontWeight.w700,
            fontSize: 13,
          ),
        ),
      ),
      Expanded(
        child: SliderTheme(
          data: SliderThemeData(
            activeTrackColor: SteerColors.teal,
            inactiveTrackColor: const Color(0xFFE4ECEB),
            thumbColor: SteerColors.teal,
            trackHeight: 6,
          ),
          child: Slider(
            value: value.clamp(min, max).toDouble(),
            min: min,
            max: max,
            onChanged: onChanged,
            onChangeEnd: (_) => onEnd(),
          ),
        ),
      ),
      SizedBox(
        width: 44,
        child: Text(
          '${(value * 100).round()}%',
          textAlign: TextAlign.right,
          style: const TextStyle(
            color: SteerColors.teal,
            fontWeight: FontWeight.w800,
            fontSize: 13,
          ),
        ),
      ),
    ],
  );
}

/// Manual control: hold FWD / BACK with one thumb and drag the steering pad
/// with the other. Driving and steering work at the same time.
class ManualControlScreen extends StatefulWidget {
  final AppState appState;
  const ManualControlScreen({super.key, required this.appState});

  @override
  State<ManualControlScreen> createState() => _ManualControlScreenState();
}

class _ManualControlScreenState extends State<ManualControlScreen> {
  AppState get appState => widget.appState;

  @override
  void initState() {
    super.initState();
    appState.setManualActive(true);
    appState.acquireStream();
  }

  @override
  void dispose() {
    appState.setManualActive(false); // also sends a stop
    appState.releaseStream();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: appState,
      builder: (context, _) {
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 16),
              child: Column(
                children: [
                  SteerPageTopBar(appState: appState, title: 'Manual control'),
                  const SizedBox(height: 12),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(24),
                      child: Container(
                        width: double.infinity,
                        color: SteerColors.cameraBg,
                        child: RobotFeedView(appState: appState),
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  ValueListenableBuilder<(int, int, int, int, int)>(
                    valueListenable: appState.ctlNotifier,
                    builder: (context, v, _) => Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 18, vertical: 12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Flexible(
                            child: Text(
                              appState.commandName,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                color: SteerColors.dark,
                                fontSize: 18,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ),
                          Text(
                            'Speed ${v.$4}%   Turn ${v.$5}%',
                            style: const TextStyle(
                              color: SteerColors.subtitle,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  SizedBox(
                    height: 170,
                    child: Row(
                      children: [
                        SizedBox(
                          width: 112,
                          child: Column(
                            children: [
                              Expanded(
                                child: _HoldButton(
                                  label: 'FORWARD',
                                  icon: Icons.arrow_upward_rounded,
                                  onHold: (down) =>
                                      appState.setManualDrive(down ? 1 : 0),
                                ),
                              ),
                              const SizedBox(height: 10),
                              Expanded(
                                child: _HoldButton(
                                  label: 'BACKWARD',
                                  icon: Icons.arrow_downward_rounded,
                                  onHold: (down) =>
                                      appState.setManualDrive(down ? -1 : 0),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _SteerPad(onSteer: appState.setManualSteer),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Column(
                      children: [
                        _miniSlider(
                          label: 'Drive speed',
                          value: appState.maxMotorSpeed,
                          min: 0.2,
                          max: 1.0,
                          onChanged: (v) => appState.updateSensitivity(speed: v),
                          onEnd: appState.saveSettings,
                        ),
                        _miniSlider(
                          label: 'Turn strength',
                          value: appState.turnStrength,
                          min: 0.2,
                          max: 1.0,
                          onChanged: (v) => appState.updateSensitivity(turn: v),
                          onEnd: appState.saveSettings,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: SteerColors.danger,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(18),
                        ),
                      ),
                      onPressed: appState.emergencyStop,
                      child: const Text(
                        'STOP',
                        style: TextStyle(
                            fontWeight: FontWeight.w800, fontSize: 17),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
