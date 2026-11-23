#pragma once
// Research-only PCM owner; not a player, protocol queue or release policy.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

class ResearchPcmHoldback {
public:
    struct Snapshot {
        size_t committed = 0;
        size_t decoded = 0;
        std::vector<float> held;
    };

    explicit ResearchPcmHoldback(size_t capacity) : capacity_(capacity) {}

    // A callback's storage is reusable scratch. Own it before returning.
    // Keep the newest callback; ordinary arrival commits the older suffix.
    // Terminal drain never advances the irreversible cursor.
    template<class Sink>
    bool append(const float * samples, size_t count, bool terminal, Sink && sink) {
        if (failed_ || closed_ || !samples || !count || count > capacity_ ||
            held_.size() > capacity_ - count) return fail();
        for (size_t i = 0; i < count; ++i) if (!std::isfinite(samples[i])) return fail();
        peak_ = std::max(peak_, held_.size() + count);
        if (!terminal && !commit(sink)) return false;
        held_.insert(held_.end(), samples, samples + count);
        return true;
    }

    Snapshot snapshot() const { return {committed_, committed_ + held_.size(), held_}; }

    // C stays monotonic. Snapshot audio covers [saved.C,D), not current C.
    bool can_restore(const Snapshot & saved) const {
        return !failed_ && !closed_ && saved.held.size() <= capacity_ &&
            saved.decoded >= saved.committed && saved.decoded - saved.committed == saved.held.size() &&
            committed_ >= saved.committed && committed_ <= saved.decoded;
    }
    bool restore(const Snapshot & saved) {
        if (!can_restore(saved)) return false;
        held_.assign(saved.held.begin() + (committed_ - saved.committed), saved.held.end());
        return true;
    }
    template<class Sink>
    bool finish(Sink && sink) {
        if (failed_ || closed_ || !commit(sink)) return false;
        closed_ = true;
        return true;
    }
    void abort() { failed_ = true; held_.clear(); }
    size_t committed() const { return committed_; }
    size_t held() const { return held_.size(); }
    size_t decoded() const { return committed_ + held_.size(); }
    size_t peak() const { return peak_; }
    bool failed() const { return failed_; }

private:
    bool fail() { failed_ = true; return false; }
    template<class Sink>
    bool commit(Sink & sink) {
        if (held_.empty()) return true;
        // A failed sink may have consumed bytes: prohibit any further retry.
        if (!sink(held_.data(), held_.size())) return fail();
        committed_ += held_.size();
        held_.clear();
        return true;
    }
    const size_t capacity_;
    size_t committed_ = 0;
    size_t peak_ = 0;
    bool failed_ = false;
    bool closed_ = false;
    std::vector<float> held_;
};
