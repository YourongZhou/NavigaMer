#ifndef NAVIGAMER_BUILD_PROGRESS_HPP
#define NAVIGAMER_BUILD_PROGRESS_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>

namespace navigamer {

class BuildProgressReporter {
 public:
  explicit BuildProgressReporter(int interval_seconds,
                                 std::ostream& output = std::cerr);
  ~BuildProgressReporter();

  BuildProgressReporter(const BuildProgressReporter&) = delete;
  BuildProgressReporter& operator=(const BuildProgressReporter&) = delete;

  void begin_phase(const std::string& phase, uint64_t total);
  void set_completed(uint64_t completed);
  void advance(uint64_t delta);
  void report_now(const char* event);
  void finish_phase();

 private:
  void reporter_loop();
  void emit_report(const char* event);

  int interval_seconds_ = 0;
  std::ostream* output_ = nullptr;
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> total_{0};

  std::mutex state_mutex_;
  std::condition_variable wakeup_;
  std::string phase_;
  std::chrono::steady_clock::time_point phase_start_;
  bool phase_active_ = false;
  bool stopping_ = false;

  std::mutex output_mutex_;
  std::thread reporter_thread_;
};

}  // namespace navigamer

#endif  // NAVIGAMER_BUILD_PROGRESS_HPP
