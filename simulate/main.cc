// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>
#include "glfw_adapter.h"
#include "simulate.h"
#include "array_safety.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
  #include <windows.h>
#else
  #if defined(__APPLE__)
    #include <mach-o/dyld.h>
  #endif
  #include <errno.h>
  #include <unistd.h>
#endif
}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncMisalign = 0.1;        // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.7;  // fraction of refresh available for simulation
const int kErrorLength = 1024;          // load error string length

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

using Seconds = std::chrono::duration<double>;


//---------------------------------------- plugin handling -----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new(std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *=2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#else
  const char* path = "/proc/self/exe";
#endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new(std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}



// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif


  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}


//------------------------------------------- simulation -------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj::Simulate::Clock::now();

  std::string filename_str(filename);
  std::string extension;
  size_t dot_pos = filename_str.rfind('.');

  if (dot_pos != std::string::npos && dot_pos < filename_str.length() - 1) {
    extension = filename_str.substr(dot_pos);
  }

  if (extension == ".mjb") {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else if (extension == ".xml") {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
  } else {
    mjSpec* spec = mj_parse(filename, nullptr, nullptr, loadError, kErrorLength);
    if (!spec) {
      mju::strcpy_arr(loadError, "could not parse model");
    } else {
      mnew = mj_compile(spec, nullptr);
      mj_deleteSpec(spec);
    }
  }

  // remove trailing newline character from loadError
  if (loadError[0]) {
    int error_length = mju::strlen_arr(loadError);
    if (loadError[error_length-1] == '\n') {
      loadError[error_length-1] = '\0';
    }
  }

  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  if (!mnew) {
    std::printf("%s\n", loadError);
    mju::strcpy_arr(sim.load_error, loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  // if no error and load took more than 1/4 seconds, report load time
  else if (load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);

  return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim synchronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger than syncMisalign
          bool misaligned =
              std::abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // inject noise
            sim.InjectNoise(sim.key);

            // run single step, let next iteration deal with timing
            mj_step(m, d);
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction/sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim)*slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise(sim.key);

              // call mj_step
              mj_step(m, d);
              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          if (sim.pause_update) {
            mju_copy(d->qacc_warmstart, d->qacc, m->nv);
          }
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}
}  // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate* sim, const char* filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
    }
    if (d) {
      sim->Load(m, d, filename);

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m, d);

    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d);
  mj_deleteModel(m);
}

//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

using MapVector3d = Eigen::Map<Eigen::Vector3d>;
using MapVectorXd = Eigen::Map<Eigen::VectorXd>;
using Isometry3d = Eigen::Transform<double, 3, Eigen::Isometry>;
using Isometry3f = Eigen::Transform<float, 3, Eigen::Isometry>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using MatrixXdRowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// inline int getBodyId(const std::string& body_name)
// {
//   int bodyid = mj_name2id(impl->m, mjOBJ_BODY, body_name.c_str());
//   if (bodyid == -1)
//     mju_error("Could not find body with name %s", body_name.c_str());
//   return bodyid;
// }

Isometry3d getBodyGlobalPose(const mjData* d, int bodyid)
{
  double qw = d->xquat[4 * bodyid + 0];
  double qx = d->xquat[4 * bodyid + 1];
  double qy = d->xquat[4 * bodyid + 2];
  double qz = d->xquat[4 * bodyid + 3];

  Eigen::Quaternion<double> quat(qw, qx, qy, qz);

  Isometry3d pose;
  pose.linear() = quat.toRotationMatrix();
  pose.translation() = MapVectorXd(&d->xpos[bodyid * 3], 3);

  return pose;
}

Isometry3d getBodyGlobalTransform(const mjData* d, int bodyid)
{
  double qw = d->xquat[4 * bodyid + 0];
  double qx = d->xquat[4 * bodyid + 1];
  double qy = d->xquat[4 * bodyid + 2];
  double qz = d->xquat[4 * bodyid + 3];

  Eigen::Quaternion<double> quat(qw, qx, qy, qz);

  Isometry3d transform;
  transform.linear() = quat.toRotationMatrix();
  transform.translation() = MapVector3d(&d->xpos[3 * bodyid]);

  return transform;
}

Eigen::VectorXd getQvel(const mjModel* m, mjData* d) { 
  return MapVectorXd(d->qvel, m->nv); 
}

constexpr char kAttrSpeed[] = "speed";

std::optional<mjtNum> ReadOptionalDoubleAttr(const mjModel* m, int instance,
                                             const char* attr) {
  const char* value = mj_getPluginConfig(m, instance, attr);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::strtod(value, nullptr);
}

class Conveyor {
  public:
  Conveyor(const mjModel* m, mjData* d, int instance);
  Conveyor(Conveyor&&) = default;
  ~Conveyor() = default;
  void Compute(const mjModel* m, mjData* d, int instance);
  void Visualize(const mjModel* m, mjData* d, mjvScene* scn, int instance);
  void Reset();
  private:
    int id_{-1}; // index of body to which plugin is attached
    std::string name_; // name of body to which plugin is attached
    mjtNum speed_{0.0}; // speed of conveyor along the y axis
};

Conveyor::Conveyor(const mjModel* m, mjData* d, int instance) {
  // Get index of body to which plugin is attached
  for (int i = 0; i < m->nbody; i++) {
    if (m->body_plugin[i] == instance) {
        id_ = i;
        break;
    }
  }
  if (id_ < 0) {
    mju_error("Could not find body attached to conveyor instance %d", instance);
  } else {
    name_ = mj_id2name(m, mjOBJ_BODY, id_);
    if (!name_.empty()) {
      mju_warning("Conveyor attached to body %s with id %d", name_.c_str(), id_);
    } else {
      mju_warning("Conveyor attached to <unknown> body with id %d", id_);
    }
  }

  // Read speed attribute
  auto maybe_speed = ReadOptionalDoubleAttr(m, instance, kAttrSpeed);
  if (maybe_speed) {
    speed_ = *maybe_speed;
  } else {
    mju_warning("Conveyor speed attribute missing, defaulting to 0.1");
    speed_ = 0.1;
  }
}

void Conveyor::Compute(const mjModel* m, mjData* d, int instance) {
  // Check valid id
  if (id_ < 0) {
    mju_error("Invalid body id %d", id_);
    return;
  }

  Vector6d spatial_force;
  const Eigen::Vector3d velocity(0.0, speed_, 0.0);

  int cid = id_;
  for (int i = 0; i < d->ncon; ++i) {
    auto& contact = d->contact[i];
    int body1 = m->geom_bodyid[contact.geom1];
    int body2 = m->geom_bodyid[contact.geom2];

    if (body1 == cid || body2 == cid) {
      Eigen::Matrix3d sim_to_body = getBodyGlobalTransform(d, cid).linear();
      Eigen::Vector3d v_conveyor = sim_to_body * velocity;

      // Normal force magnitude
      mj_contactForce(m, d, i, spatial_force.data());
      double N =
        spatial_force[0]; // According to mujoco's documentation, normal is defined as the x axis

      // Body jacobian
      MatrixXdRowMajor J_body(3, m->nv);
      if (body1 != cid)
        mj_jac(m, d, J_body.data(), nullptr, contact.pos, body1);
      else
        mj_jac(m, d, J_body.data(), nullptr, contact.pos, body2);

      // Body velocity at contact point
      Eigen::Vector3d v_body = J_body * getQvel(m, d);

      // Adjust frictional force along the conveyor axis
      // Get contact point velocity along the conveyor motion
      double v_rel = (v_body - v_conveyor).dot(v_conveyor.normalized());
      if (v_rel > 0) {
        // Body point faster than conveyor so let the friction from the static surface effect
        // normally
        return;
      }
      if (v_body.dot(v_conveyor.normalized()) < 0) {
        // Body point moving in the opposite direction, friction from the static surface is already
        // been applied
        return;
      }
      // Compensate for the friction of the static surface + add friction corresponding to moving
      // conveyor
      Eigen::Vector3d friction_force = N * (2 * contact.friction[0]) * v_conveyor.normalized();

      // Calculate conveyor frictional force in generalized coordinates
      Eigen::VectorXd qfrc = J_body.transpose() * friction_force;

      // Add contact force
      MapVectorXd(d->qfrc_passive, m->nv) += qfrc;
    }
  }

}

void Conveyor::Reset() {

}

void registerPlugins() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.custom.conveyor";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;

  const char* attributes[] = {"speed"};
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.attributes = attributes;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto* conveyor = new Conveyor(m, d, instance);
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(conveyor);
  return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Conveyor*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int type) {
        auto conveyor = reinterpret_cast<Conveyor*>(d->plugin_data[instance]);
        conveyor->Compute(m, d, instance);
      };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn,
                         int instance) {
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
    int instance) {
    auto conveyor = reinterpret_cast<Conveyor*>(plugin_data);
    conveyor->Reset();
};
  mjp_registerPlugin(&plugin);
}

// run event loop
int main(int argc, char** argv) {

  registerPlugins();

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER!=mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
      std::make_unique<mj::GlfwAdapter>(),
      &cam, &opt, &pert, /* is_passive = */ false
  );

  const char* filename = nullptr;
  if (argc >  1) {
    filename = argv[1];
  }

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename);

  // start simulation UI loop (blocking call)
  sim->RenderLoop();
  physicsthreadhandle.join();

  return 0;
}
