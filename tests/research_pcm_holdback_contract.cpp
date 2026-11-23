#include "research_pcm_holdback.h"
#include <limits>
#include <stdexcept>

namespace {
void require(bool value) { if (!value) throw std::runtime_error("PCM holdback contract failed"); }
}
int main() {
    std::vector<float> committed;
    auto sink = [&](const float * p, size_t n) { committed.insert(committed.end(), p, p + n); return true; };
    ResearchPcmHoldback buffer(16);
    float scratch[] = {1, 2};
    require(buffer.append(scratch, 2, false, sink));
    const auto at2 = buffer.snapshot();
    scratch[0] = 9;
    require(buffer.append(scratch, 2, false, sink));
    require(committed == std::vector<float>({1, 2}));
    require(buffer.committed() == 2 && buffer.restore(at2) && buffer.committed() == 2 && buffer.held() == 0);
    require(buffer.append(scratch, 2, true, sink));
    require(buffer.append(scratch, 2, true, sink));
    require(buffer.committed() == 2 && buffer.held() == 4);
    require(buffer.finish(sink) && committed == std::vector<float>({1, 2, 9, 2, 9, 2}));
    require(!buffer.restore(at2) && !buffer.finish(sink));

    ResearchPcmHoldback partial(16);
    require(partial.append(scratch, 2, false, sink));
    const auto old = partial.snapshot();
    require(partial.append(scratch, 2, false, sink));
    require(partial.append(scratch, 2, false, sink));
    require(partial.committed() == 4 && !partial.can_restore(old) && !partial.restore(old));
    require(partial.held() == 2 && partial.committed() == 4);

    // A checkpoint can contain a prefix committed after it was captured.
    ResearchPcmHoldback sliced(16);
    require(sliced.append(scratch, 2, true, sink));
    require(sliced.append(scratch, 2, true, sink));
    const auto saved4 = sliced.snapshot();
    auto saved_bad = saved4;
    saved_bad.decoded++;
    require(!sliced.restore(saved_bad));
    require(sliced.append(scratch, 2, false, sink));
    require(sliced.committed() == 4 && sliced.restore(saved4) && sliced.held() == 0);
    ResearchPcmHoldback inner(16);
    require(inner.append(scratch, 2, false, sink));
    require(inner.append(scratch, 2, false, sink));
    require(inner.committed() == 2 && inner.restore(saved4) && inner.held() == 2);
    require(inner.snapshot().held == std::vector<float>({9, 2}));

    ResearchPcmHoldback cap(3);
    const auto before = cap.snapshot();
    require(cap.append(scratch, 2, true, sink));
    require(!cap.append(scratch, 2, true, sink) && cap.failed() && cap.held() == 2);
    require(!cap.restore(before));
    ResearchPcmHoldback invalid(4);
    scratch[0] = std::numeric_limits<float>::quiet_NaN();
    require(!invalid.append(scratch, 2, false, sink) && invalid.failed());
    scratch[0] = 1;

    ResearchPcmHoldback failed_sink(8);
    auto refusing = [](const float *, size_t) { return false; };
    require(failed_sink.append(scratch, 2, false, sink));
    const auto sf = failed_sink.snapshot();
    require(!failed_sink.append(scratch, 2, false, refusing) && failed_sink.failed());
    require(!failed_sink.restore(sf) && !failed_sink.finish(sink));

    // Cancellation/deadline owner aborts before finish or restoring audio.
    for (int reason = 0; reason < 2; ++reason) {
        ResearchPcmHoldback aborted(8);
        require(aborted.append(scratch, 2, false, sink));
        const auto checkpoint = aborted.snapshot();
        const auto count = committed.size();
        aborted.abort();
        require(!aborted.restore(checkpoint) && !aborted.finish(sink) && committed.size() == count);
    }
}
