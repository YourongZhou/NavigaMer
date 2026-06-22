#include "build_progress.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace navigamer {

namespace {

std::string timestamp_now() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t value = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
#if defined(_WIN32)
  localtime_s(&local, &value);
#else
  localtime_r(&value, &local);
#endif
  std::ostringstream out;
  out << std::put_time(&local, "%Y-%m-%dT%H:%M:%S%z");
  return out.str();
}

}  // namespace

BuildProgressReporter::BuildProgressReporter(int interval_seconds,
                                             std::ostream& output)
    : interval_seconds_(interval_seconds), output_(&output) {
  if (interval_seconds < 0) {
    throw std::invalid_argument("progress interval must be nonnegative");
  }
  if (interval_seconds_ > 0) {
    reporter_thread_ = std::thread(&BuildProgressReporter::reporter_loop, this);
  }
}

BuildProgressReporter::~BuildProgressReporter() {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    stopping_ = true;
  }
  wakeup_.notify_all();
  if (reporter_thread_.joinable()) reporter_thread_.join();
}

void BuildProgressReporter::begin_phase(const std::string& phase,
                                        uint64_t total) {
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    completed_.store(0, std::memory_order_relaxed);
    total_.store(total, std::memory_order_relaxed);
    phase_ = phase;
    phase_start_ = std::chrono::steady_clock::now();
    phase_active_ = true;
  }
  emit_report("start");
}

void BuildProgressReporter::set_completed(uint64_t completed) {
  completed_.store(completed, std::memory_order_relaxed);
}

void BuildProgressReporter::advance(uint64_t delta) {
  completed_.fetch_add(delta, std::memory_order_relaxed);
}

void BuildProgressReporter::report_now(const char* event) {
  emit_report(event);
}

void BuildProgressReporter::finish_phase() {
  const uint64_t total = total_.load(std::memory_order_relaxed);
  if (total > 0) completed_.store(total, std::memory_order_relaxed);
  emit_report("finish");
  std::lock_guard<std::mutex> lock(state_mutex_);
  phase_active_ = false;
}

void BuildProgressReporter::reporter_loop() {
  std::unique_lock<std::mutex> lock(state_mutex_);
  while (!stopping_) {
    const bool stopped = wakeup_.wait_for(
        lock, std::chrono::seconds(interval_seconds_),
        [this] { return stopping_; });
    if (stopped) break;
    const bool active = phase_active_;
    lock.unlock();
    if (active) emit_report("heartbeat");
    lock.lock();
  }
}

void BuildProgressReporter::emit_report(const char* event) {
  std::string phase;
  std::chrono::steady_clock::time_point phase_start;
  uint64_t completed = 0;
  uint64_t total = 0;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!phase_active_) return;
    phase = phase_;
    phase_start = phase_start_;
    completed = completed_.load(std::memory_order_relaxed);
    total = total_.load(std::memory_order_relaxed);
  }

  const double elapsed_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    phase_start)
          .count();
  const double rate = elapsed_seconds > 0.0
                          ? static_cast<double>(completed) / elapsed_seconds
                          : 0.0;

  std::ostringstream line;
  line << "Build progress: timestamp=" << timestamp_now()
       << " event=" << event
       << " phase=" << phase
       << " completed=" << completed
       << " total=" << total
       << std::fixed << std::setprecision(1)
       << " percent="
       << (total > 0 ? 100.0 * static_cast<double>(completed) /
                            static_cast<double>(total)
                     : 0.0)
       << " elapsed_s=" << elapsed_seconds
       << " rate_per_s=" << rate
       << " eta_s=";
  if (rate > 0.0 && total > completed) {
    line << static_cast<double>(total - completed) / rate;
  } else if (total <= completed) {
    line << 0.0;
  } else {
    line << "unknown";
  }

  std::lock_guard<std::mutex> lock(output_mutex_);
  (*output_) << line.str() << '\n';
  output_->flush();
}

}  // namespace navigamer
