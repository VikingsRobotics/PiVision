// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <array>
#include <type_traits>

#include <wpi/Endian.h>
#include <fmt/format.h>
#include <networktables/NetworkTableInstance.h>
#include <vision/VisionPipeline.h>
#include <vision/VisionRunner.h>
#include <wpi/StringExtras.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>

#include <networktables/NetworkTableInstance.h>
#include <networktables/RawTopic.h>

#include <frc/apriltag/AprilTagDetector.h>
#include <frc/apriltag/AprilTagPoseEstimator.h>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cameraserver/CameraServer.h"

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
       "switched cameras": [
           {
               "name": <virtual camera name>
               "key": <network table key used for selection>
               // if NT value is a string, it's treated as a name
               // if NT value is a double, it's treated as an integer index
           }
       ]
   }
 */

static const char* configFile = "/boot/frc.json";

namespace {

unsigned int team;
bool server = false;

struct CameraConfig {
  std::string name;
  std::string path;
  wpi::json config;
  wpi::json streamConfig;
};

struct SwitchedCameraConfig {
  std::string name;
  std::string key;
};

std::vector<CameraConfig> cameraConfigs;
std::vector<SwitchedCameraConfig> switchedCameraConfigs;
std::vector<cs::VideoSource> cameras;

void ParseErrorV(fmt::string_view format, fmt::format_args args) {
  fmt::print(stderr, "config error in '{}': ", configFile);
  fmt::vprint(stderr, format, args);
  fmt::print(stderr, "\n");
}

template <typename... Args>
void ParseError(fmt::string_view format, Args&&... args) {
  ParseErrorV(format, fmt::make_format_args(args...));
}

bool ReadCameraConfig(const wpi::json& config) {
  CameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError("could not read camera name: {}", e.what());
    return false;
  }

  // path
  try {
    c.path = config.at("path").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError("camera '{}': could not read path: {}", c.name, e.what());
    return false;
  }

  // stream properties
  if (config.count("stream") != 0) c.streamConfig = config.at("stream");

  c.config = config;

  cameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadSwitchedCameraConfig(const wpi::json& config) {
  SwitchedCameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError("could not read switched camera name: {}", e.what());
    return false;
  }

  // key
  try {
    c.key = config.at("key").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError("switched camera '{}': could not read key: {}", c.name,
               e.what());
    return false;
  }

  switchedCameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadConfig() {
  // open config file
  std::error_code ec;
  wpi::raw_fd_istream is(configFile, ec);
  if (ec) {
    fmt::print(stderr, "could not open '{}': {}", configFile, ec.message());
    return false;
  }

  // parse file
  wpi::json j;
  try {
    j = wpi::json::parse(is);
  } catch (const wpi::json::parse_error& e) {
    ParseError("byte {}: {}", e.byte, e.what());
    return false;
  }

  // top level must be an object
  if (!j.is_object()) {
    ParseError("must be JSON object");
    return false;
  }

  // team number
  try {
    team = j.at("team").get<unsigned int>();
  } catch (const wpi::json::exception& e) {
    ParseError("could not read team number: {}", e.what());
    return false;
  }

  // ntmode (optional)
  if (j.count("ntmode") != 0) {
    try {
      auto str = j.at("ntmode").get<std::string>();
      if (wpi::equals_lower(str, "client")) {
        server = false;
      } else if (wpi::equals_lower(str, "server")) {
        server = true;
      } else {
        ParseError("could not understand ntmode value '{}'", str);
      }
    } catch (const wpi::json::exception& e) {
      ParseError("could not read ntmode: {}", e.what());
    }
  }

  // cameras
  try {
    for (auto&& camera : j.at("cameras")) {
      if (!ReadCameraConfig(camera)) return false;
    }
  } catch (const wpi::json::exception& e) {
    ParseError("could not read cameras: {}", e.what());
    return false;
  }

  // switched cameras (optional)
  if (j.count("switched cameras") != 0) {
    try {
      for (auto&& camera : j.at("switched cameras")) {
        if (!ReadSwitchedCameraConfig(camera)) return false;
      }
    } catch (const wpi::json::exception& e) {
      ParseError("could not read switched cameras: {}", e.what());
      return false;
    }
  }

  return true;
}

cs::UsbCamera StartCamera(const CameraConfig& config) {
  fmt::print("Starting camera '{}' on {}\n", config.name, config.path);
  cs::UsbCamera camera{config.name, config.path};
  auto server = frc::CameraServer::StartAutomaticCapture(camera);

  camera.SetConfigJson(config.config);
  camera.SetConnectionStrategy(cs::VideoSource::kConnectionKeepOpen);

  if (config.streamConfig.is_object())
    server.SetConfigJson(config.streamConfig);

  return camera;
}

cs::MjpegServer StartSwitchedCamera(const SwitchedCameraConfig& config) {
  fmt::print("Starting switched camera '{}' on {}\n", config.name, config.key);
  auto server = frc::CameraServer::AddSwitchedCamera(config.name);

  auto inst = nt::NetworkTableInstance::GetDefault();
  inst.AddListener(
    inst.GetTopic(config.key),
    nt::EventFlags::kImmediate | nt::EventFlags::kValueAll,
    [server](const auto& event) mutable {
      if (auto data = event.GetValueEventData()) {
        if (data->value.IsInteger()) {
          int i = data->value.GetInteger();
          if (i >= 0 && i < cameras.size()) server.SetSource(cameras[i]);
        } else if (data->value.IsDouble()) {
          int i = data->value.GetDouble();
          if (i >= 0 && i < cameras.size()) server.SetSource(cameras[i]);
        } else if (data->value.IsString()) {
          auto str = data->value.GetString();
          for (int i = 0; i < cameraConfigs.size(); ++i) {
            if (str == cameraConfigs[i].name) {
              server.SetSource(cameras[i]);
              break;
            }
          }
        }
      }
    });

  return server;
}

// example pipeline
class MyPipeline : public frc::VisionPipeline {
public:
  MyPipeline(frc::AprilTagPoseEstimator::Config estimatorConfig,
    frc::AprilTagDetector::Config detectorConfig,
    frc::AprilTagDetector::QuadThresholdParameters detectorThres) : 
    estimator{estimatorConfig} {
      detector.SetConfig(detectorConfig);
      detector.SetQuadThresholdParameters(detectorThres);
      detector.AddFamily("tag36h11",2);
      tags.reserve(30);
  }
  
  frc::AprilTagDetector detector { };
  frc::AprilTagPoseEstimator estimator;
  cv::Mat grayMat;

  cs::CvSource outputStream =
      frc::CameraServer::PutVideo("Detected", 120, 90);

  struct DetectedResults {
    int tagId;
    float confidence; 
    frc::Transform3d pose;
  };

  std::vector<DetectedResults> tags;
  cv::Scalar outlineColor{ 0, 255, 0 };
  cv::Scalar crossColor{ 0, 0, 255 };

  void Process(cv::Mat& mat) override {
      cv::cvtColor(mat, grayMat, cv::COLOR_BGR2GRAY);

      cv::Size g_size = grayMat.size();
      frc::AprilTagDetector::Results detections =
          detector.Detect(g_size.width, g_size.height, grayMat.data);

      tags.clear();

      for (const frc::AprilTagDetection* detection : detections) {
          // draw lines around the tag
          for (int i = 0; i <= 3; i++) {
              int j = (i + 1) % 4;
              const frc::AprilTagDetection::Point pti = detection->GetCorner(i);
              const frc::AprilTagDetection::Point ptj = detection->GetCorner(j);
              line(mat, cv::Point(pti.x, pti.y), cv::Point(ptj.x, ptj.y),
                  outlineColor, 2);
          }
          
          // mark the center of the tag
          const frc::AprilTagDetection::Point c = detection->GetCenter();
          int ll = 10;
          line(mat, cv::Point(c.x - ll, c.y), cv::Point(c.x + ll, c.y),
              crossColor, 2);
          line(mat, cv::Point(c.x, c.y - ll), cv::Point(c.x, c.y + ll),
              crossColor, 2);

          // identify the tag
          putText(mat, std::to_string(detection->GetId()),
              cv::Point(c.x + ll, c.y), cv::FONT_HERSHEY_SIMPLEX, 1,
              crossColor, 3);

          // remember we saw this tag and pose
          tags.emplace_back(
            DetectedResults {
              .tagId = detection->GetId(),
              .confidence = detection->GetDecisionMargin(),
              .pose = estimator.Estimate(*detection)
            }
          );
      }

      outputStream.PutFrame(mat);
  }
};
}  // namespace

template <class T2, class T1>
T2 bit_cast(T1 t1) {
  static_assert(sizeof(T1)==sizeof(T2), "Types must match sizes");
  static_assert(std::is_trivial<T1>::value, "Requires POD input");
  static_assert(std::is_trivial<T2>::value, "Requires POD output");

  T2 t2;
  std::memcpy( std::addressof(t2), std::addressof(t1), sizeof(T1) );
  return t2;
}

int main(int argc, char* argv[]) {
  if (argc >= 2) configFile = argv[1];

  // read configuration
  if (!ReadConfig()) return EXIT_FAILURE;

  // start NetworkTables
  auto ntinst = nt::NetworkTableInstance::GetDefault();
  if (server) {
    fmt::print("Setting up NetworkTables server\n");
    ntinst.StartServer();
  } else {
    fmt::print("Setting up NetworkTables client for team {}\n", team);
    ntinst.StartClient4("multiCameraServer");
    ntinst.SetServerTeam(team);
    ntinst.StartDSClient();
  }

  // start cameras
  // work around wpilibsuite/allwpilib#5055
  frc::CameraServer::SetSize(frc::CameraServer::kSize640x480);
  for (const auto& config : cameraConfigs)
    cameras.emplace_back(StartCamera(config));

  // start switched cameras
  for (const auto& config : switchedCameraConfigs) StartSwitchedCamera(config);

  // start image processing on camera 0 if present
  if (cameras.size() >= 1) {
    std::thread([&] {
      auto tagsTable = ntinst.GetTable("april-tag");
      nt::RawPublisher tagsPublisher = tagsTable->GetRawTopic("tag").Publish("AprilTagWithConfidence");
      std::array<uint8_t,4> defaultPack { 0, 0, 0, 0};
      tagsPublisher.SetDefault(std::span<uint8_t,4>{defaultPack});
      MyPipeline camPipeline {
        frc::AprilTagPoseEstimator::Config{
          .tagSize = 6.5_in,
          .fx = 699.3778103158814,
          .fy = 677.7161226393544,
          .cx = 345.6059345433618,
          .cy = 207.12741326228522
        },
        frc::AprilTagDetector::Config{ 
          .numThreads = 2,
          .quadDecimate = 2.0,
          .quadSigma = 0.1,
          .refineEdges = true,
          .decodeSharpening = 0.5,
          .debug = false
        },
        frc::AprilTagDetector::QuadThresholdParameters{
          .minClusterPixels = 5,
          .maxNumMaxima = 10,
          .criticalAngle = 10_deg,
          .maxLineFitMSE = 10.0,
          .minWhiteBlackDiff = 5,
          .deglitch = false
        }
      };
      frc::VisionRunner<MyPipeline> runner(cameras[0], &camPipeline,
      [&](MyPipeline &pipeline) {
        using namespace wpi::support::endian;
        static std::array<uint8_t,1236> data = { 0 };
        static uint8_t times = 0;
        uint8_t numOfFoundTags = pipeline.tags.size();
        times = ++times & 0b01111111;

        data[0] = numOfFoundTags;
        data[1] = 0b10000000 | times;
        data[2] = 0;
        data[3] = 0;

        constexpr uint8_t * at = data.data() + 4;

        for (size_t index = 0; index < numOfFoundTags; ++index)
        {
          write32be(at + (0 + 56 * index),bit_cast<uint32_t>(pipeline.tags.at(index).confidence));
          write32be(at + (4 + 56 * index),bit_cast<uint32_t>(pipeline.tags.at(index).tagId));
          const frc::Transform3d& pose = pipeline.tags.at(index).pose;
          const frc::Translation3d translation = pose.Translation();
          write64be(at + (8 + 56 * index),bit_cast<uint64_t>(translation.X().value()));
          write64be(at + (16 + 56 * index),bit_cast<uint64_t>(translation.Y().value()));
          write64be(at + (24 + 56 * index),bit_cast<uint64_t>(translation.Z().value()));
          const frc::Rotation3d rotation = pose.Rotation();
          write64be(at + (32 + 56 * index),bit_cast<uint64_t>(rotation.X().value()));
          write64be(at + (40 + 56 * index),bit_cast<uint64_t>(rotation.Y().value()));
          write64be(at + (48 + 56 * index),bit_cast<uint64_t>(rotation.Z().value()));
        }
        tagsPublisher.Set(std::span<uint8_t>{data.begin(), 4 + (56 * static_cast<size_t>(numOfFoundTags))});
      });
      runner.RunForever();
    }).detach();
  }

  // loop forever
  for (;;) std::this_thread::sleep_for(std::chrono::seconds(10));
}
