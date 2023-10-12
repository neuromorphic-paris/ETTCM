/**
 * @file
 * @brief Global decay estimator implementation.
 */

#ifndef ETTCM_GLOBAL_DECAY_HPP
#define ETTCM_GLOBAL_DECAY_HPP

#include <cstdint>
#include <utility>

#include "ETTCM/types.hpp"
#include "ETTCM/utils.hpp"

namespace ETTCM
{
/**
 * @brief Global decay estimator.
 *
 * This class estimates the decay from a stream of events generated by a single
 * motion.
 *
 * @tparam Event Type of event.
 * @tparam EventToDecay Type of the handle to pass from an event to a decay.
 * @tparam HandleDecay Type of the handle to further process the estimated
 * decay.
 */
template <typename Event, typename EventToDecay, typename HandleDecay>
class GlobalDecay
{
 public:
  /**
   * @brief Constructs an instance to estimate the global decay from a stream of
   * events.
   *
   * @param t_decay_first @copybrief t_decay_first_
   * @param event_to_decay @copybrief event_to_decay_
   * @param handle_decay @copybrief handle_decay_
   */
  GlobalDecay(const uint64_t t_decay_first, EventToDecay&& event_to_decay,
              HandleDecay&& handle_decay)
      : t_decay_first_(t_decay_first),
        event_to_decay_(std::forward<EventToDecay>(event_to_decay)),
        handle_decay_(std::forward<HandleDecay>(handle_decay))
  {
    reset();
  }
  /**
   * @brief Deleted copy constructor.
   */
  GlobalDecay(const GlobalDecay&) = delete;
  /**
   * @brief Default move constructor.
   */
  GlobalDecay(GlobalDecay&&) = default;
  /**
   * @brief Deleted copy assignment operator.
   */
  GlobalDecay&
  operator=(const GlobalDecay&) = delete;
  /**
   * @brief Default move assignment operator.
   */
  GlobalDecay&
  operator=(GlobalDecay&&) = default;
  /**
   * @brief Default destructor.
   */
  ~GlobalDecay() = default;

  /**
   * @brief Returns the current timestamp \f$[\text{microseconds}]\f$.
   *
   * @return Current timestamp \f$[\text{microseconds}]\f$.
   */
  uint64_t
  t() const
  {
    return decay_.t;
  }

  /**
   * @brief Returns the current decay.
   *
   * @return Current decay.
   */
  float
  decay() const
  {
    return decay_.decay;
  }

  /**
   * @brief Returns the count of the incoming number of events.
   *
   * @return Count of the incoming number of events.
   */
  float
  n_decay() const
  {
    return decay_.n_decay;
  }

  /**
   * @brief Returns the event time decay \f$[\text{microseconds}]\f$.
   *
   * @return Event time decay \f$[\text{microseconds}]\f$.
   */
  float
  t_decay() const
  {
    return decay_.t_decay;
  }

  /**
   * @brief Returns the current event rate
   * \f$[\text{events}/\text{microseconds}]\f$.
   *
   * @return Current event rate \f$[\text{events}/\text{microseconds}]\f$.
   */
  float
  rate() const
  {
    return decay_.rate;
  }

  /**
   * @brief Estimates the global decay one event at a time.
   *
   * This method estimates the decay in an event-by-event basis by estimating
   * the event stream activity.
   *
   * @param event Incoming event.
   */
  void
  operator()(Event event)
  {
    decay_.decay = static_cast<float>(1);
    const float t_diff =
        (event.t > decay_.t) ? static_cast<float>(event.t - decay_.t) : 0;
    if (t_diff > 0)
    {
      decay_.decay /= static_cast<float>(1e-6) * t_diff * decay_.n_decay +
                      static_cast<float>(1);

      decay_.n_decay *= decay_.decay;
      decay_.t_decay = decay_.decay * decay_.t_decay + t_diff;

      decay_.t = event.t;
    }
    ++decay_.n_decay;

    decay_.rate = decay_.n_decay / decay_.t_decay;

    handle_decay_(event_to_decay_(event, decay_.decay, decay_.n_decay,
                                  decay_.t_decay, decay_.rate));
  }

  /**
   * @brief Resets the context.
   */
  void
  reset()
  {
    decay_.reset(t_decay_first_);
  }

 protected:
  /**
   * @brief Initial time rate assumption to bootstrap the rate estimator
   * \f$[\text{microseconds}]\f$.
   */
  const uint64_t t_decay_first_;

  /**
   * @brief Decay stucture.
   * \sa ETTCM::Decay.
   */
  Decay decay_;

  /**
   * @brief Handle to pass from an event to a decay.
   */
  EventToDecay event_to_decay_;
  /**
   * @brief Handle to further process the estimated decay.
   */
  HandleDecay handle_decay_;
};

/**
 * @brief Make function that creates an instance of ETTCM::GlobalDecay.
 *
 * @tparam Event Type of event.
 * @tparam EventToDecay Type of the handle to pass from an event to a decay.
 * @tparam HandleDecay Type of the handle to further process the estimated
 * decay.
 *
 * @param t_decay_first Initial decay assumption to bootstrap the rate
 * estimator \f$[\text{microseconds}]\f$.
 * @param event_to_decay Handle to pass from an event to to a decay.
 * @param handle_decay Handle to further process the estimated decay.
 *
 * @return Instance of ETTCM::GlobalDecay.
 */
template <typename Event, typename EventToDecay, typename HandleDecay>
inline GlobalDecay<Event, EventToDecay, HandleDecay>
make_global_decay(const uint64_t t_decay_first, EventToDecay&& event_to_decay,
                  HandleDecay&& handle_decay)
{
  return GlobalDecay<Event, EventToDecay, HandleDecay>(
      t_decay_first, std::forward<EventToDecay>(event_to_decay),
      std::forward<HandleDecay>(handle_decay));
}
}  // namespace ETTCM

#endif  // ETTCM_GLOBAL_DECAY_HPP
