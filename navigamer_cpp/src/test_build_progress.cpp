#include "build_progress.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

void assert_contains(const std::string& text, const std::string& expected) {
  assert(text.find(expected) != std::string::npos);
}

}  // namespace

int main() {
  {
    std::ostringstream output;
    navigamer::BuildProgressReporter reporter(0, output);
    reporter.begin_phase("phase1_sketch", 100);
    reporter.set_completed(50);
    reporter.report_now("heartbeat");
    reporter.finish_phase();

    const std::string text = output.str();
    assert_contains(text, "timestamp=");
    assert_contains(text, "event=start");
    assert_contains(text, "event=heartbeat");
    assert_contains(text, "event=finish");
    assert_contains(text, "phase=phase1_sketch");
    assert_contains(text, "completed=50");
    assert_contains(text, "total=100");
    assert_contains(text, "percent=50.0");
    assert_contains(text, "elapsed_s=");
    assert_contains(text, "rate_per_s=");
    assert_contains(text, "eta_s=");
  }

  {
    std::ostringstream output;
    {
      navigamer::BuildProgressReporter reporter(1, output);
      reporter.begin_phase("phase2_rebinding", 10);
      reporter.set_completed(3);
      std::this_thread::sleep_for(std::chrono::milliseconds(1250));
    }
    const std::string text = output.str();
    assert_contains(text, "event=heartbeat");
    assert_contains(text, "phase=phase2_rebinding");
    assert_contains(text, "completed=3");
  }

  std::cout << "build progress tests passed\n";
  return 0;
}
