/**
 * @file
 * @brief Sampling by leaky-firing implementation.
 */

#ifndef ETTCM_LEAKY_SAMPLING_HPP
#define ETTCM_LEAKY_SAMPLING_HPP

#include <cstdint>
#include <utility>

#include "ETTCM/assert.hpp"
#include "ETTCM/types.hpp"
#include "ETTCM/utils.hpp"

namespace ETTCM
{
/**
 * @brief Sampling by leaky-firing.
 *
 * This class samples events based on a simple leaky-firing model.
 *
 * @tparam Event Type of event.
 * @tparam EventToLeakyEvent Type of the handle that passes from an event to a
 * leaky event.
 * @tparam HandleLeakyEvent Type of the handle to further process the leaky
 * events.
 */
template <typename Event, typename EventToLeakyEvent, typename HandleLeakyEvent>
class LeakySampling
{
 public:
  /**
   * @brief Type of the leaky-firing context.
   */
  typedef Matrix<float> Context;

  /**
   * @brief Constructs an instance to sample events by a simple leaky-firing
   * model.
   *
   * @param width Original context width.
   * @param height Original context height.
   * @param width_sampling_factor @copybrief width_sampling_factor_
   * @param height_sampling_factor @copybrief height_sampling_factor_
   * @param sampling_factor @copybrief sampling_factor_
   * @param event_to_leaky_event @copybrief event_to_leaky_event_
   * @param handle_leaky_event @copybrief handle_leaky_event_
   */
  LeakySampling(const uint16_t width, const uint16_t height,
                const float width_sampling_factor,
                const float height_sampling_factor, const float sampling_factor,
                EventToLeakyEvent&& event_to_leaky_event,
                HandleLeakyEvent&& handle_leaky_event)
      : width_sampling_factor_(width_sampling_factor),
        height_sampling_factor_(height_sampling_factor),
        sampling_factor_(sampling_factor),
        width_(std::ceil(static_cast<float>(width) / width_sampling_factor_)),
        height_(
            std::ceil(static_cast<float>(height) / height_sampling_factor_)),
        context_on_(Context::Zero(width_, height_)),
        context_off_(Context::Zero(width_, height_)),
        event_to_leaky_event_(
            std::forward<EventToLeakyEvent>(event_to_leaky_event)),
        handle_leaky_event_(std::forward<HandleLeakyEvent>(handle_leaky_event))
  {
    ASSERT(1 <= width_sampling_factor_,
           "The width sampling reduction factor "
               << width_sampling_factor_ << " must be greater or equal to 1");
    ASSERT(1 <= height_sampling_factor_,
           "The height sampling reduction factor "
               << height_sampling_factor_ << " must be greater or equal to 1");
  }
  /**
   * @brief Deleted copy constructor.
   */
  LeakySampling(const LeakySampling&) = delete;
  /**
   * @brief Default move constructor.
   */
  LeakySampling(LeakySampling&&) = default;
  /**
   * @brief Deleted copy assignment operator.
   */
  LeakySampling&
  operator=(const LeakySampling&) = delete;
  /**
   * @brief Default move assignment operator.
   */
  LeakySampling&
  operator=(LeakySampling&&) = default;
  /**
   * @brief Default destructor.
   */
  ~LeakySampling() = default;

  /**
   * @brief Returns the width of the leaky-firing context.
   *
   * @return Width of the context.
   */
  uint16_t
  width() const
  {
    return width_;
  }
  /**
   * @brief Returns the height of the leaky-firing context.
   *
   * @return Height of the context.
   */
  uint16_t
  height() const
  {
    return height_;
  }

  /**
   * @brief Samples incoming events based on a simple leaky-firing model.
   *
   * This method samples events based on a simple leaky-firing model in an
   * event-by-event basis.
   *
   * @param event Incoming event.
   */
  void
  operator()(Event event)
  {
    const uint16_t x = event.x / width_sampling_factor_;
    ASSERT(x < width_, "The event x-leaky coordinate "
                           << x << " must be smaller than " << width_);
    const uint16_t y = event.y / height_sampling_factor_;
    ASSERT(y < height_, "The event y-leaky coordinate "
                            << y << " must be smaller than " << height_);

    Context& context = (event.is_increase) ? context_on_ : context_off_;
    ++context(x, y);
    if (context(x, y) >= sampling_factor_)
    {
      context(x, y) -= sampling_factor_;
      handle_leaky_event_(event_to_leaky_event_(event, x, y));
    }
  }

  /**
   * @brief Resets the context.
   */
  void
  reset()
  {
    context_on_.setZero();
    context_off_.setZero();
  }

 protected:
  /**
   * @brief Width sampling factor.
   */
  const float width_sampling_factor_;
  /**
   * @brief Height sampling factor.
   */
  const float height_sampling_factor_;
  /**
   * @brief Sampling factor.
   */
  const float sampling_factor_;

  /**
   * @brief Width of the leaky-firing context computed as the quotient of the
   * original width of the context and width_sampling_factor_.
   */
  const uint16_t width_;
  /**
   * @brief Height of the leaky-firing context computed as the quotient of the
   * original height of the context and height_sampling_factor_.
   */
  const uint16_t height_;

  /**
   * @brief Leaky-firing context of events with ON polarity.
   */
  Context context_on_;
  /**
   * @brief Leaky-firing context of events with OFF polarity.
   */
  Context context_off_;

  /**
   * @brief Handle that passes from an event to a leaky event.
   */
  EventToLeakyEvent event_to_leaky_event_;
  /**
   * @brief Handle to further process the leaky event.
   */
  HandleLeakyEvent handle_leaky_event_;
};

/**
 * @brief Make function that creates an instance of
 * ETTCM::LeakySampling.
 *
 * @tparam Event Type of event.
 * @tparam EventToLeakyEvent Type of the handle that passes from an event a
 * leaky event.
 * @tparam HandleLeakyEvent Type of the handle to further process the leaky
 * events.
 *
 * @param width Original context width.
 * @param height Original context height.
 * @param width_sampling_factor Width sampling factor.
 * @param height_sampling_factor Height sampling factor.
 * @param event_to_leaky_event Handle that passes from an event to a leaky
 * event.
 * @param handle_leaky_event Handle to further process the leaky event.
 *
 * @return Instance of ETTCM::LeakySampling.
 */
template <typename Event, typename EventToLeakyEvent, typename HandleLeakyEvent>
inline LeakySampling<Event, EventToLeakyEvent, HandleLeakyEvent>
make_leaky_sampling(const uint16_t width, const uint16_t height,
                    const float width_sampling_factor,
                    const float height_sampling_factor,
                    EventToLeakyEvent&& event_to_leaky_event,
                    HandleLeakyEvent&& handle_leaky_event)
{
  return LeakySampling<Event, EventToLeakyEvent, HandleLeakyEvent>(
      width, height, width_sampling_factor, height_sampling_factor,
      width_sampling_factor * height_sampling_factor,
      std::forward<EventToLeakyEvent>(event_to_leaky_event),
      std::forward<HandleLeakyEvent>(handle_leaky_event));
}

/**
 * @brief Make function that creates an instance of
 * ETTCM::LeakySampling.
 *
 * @tparam Event Type of event.
 * @tparam EventToLeakyEvent Type of the handle that passes from an event a
 * leaky event.
 * @tparam HandleLeakyEvent Type of the handle to further process the leaky
 * events.
 *
 * @param width Original context width.
 * @param height Original context height.
 * @param width_sampling_factor Width sampling factor.
 * @param height_sampling_factor Height sampling factor.
 * @param sampling_factor Sampling factor.
 * @param event_to_leaky_event Handle that passes from an event to a leaky
 * event.
 * @param handle_leaky_event Handle to further process the leaky event.
 *
 * @return Instance of ETTCM::LeakySampling.
 */
template <typename Event, typename EventToLeakyEvent, typename HandleLeakyEvent>
inline LeakySampling<Event, EventToLeakyEvent, HandleLeakyEvent>
make_leaky_sampling(const uint16_t width, const uint16_t height,
                    const float width_sampling_factor,
                    const float height_sampling_factor,
                    const float sampling_factor,
                    EventToLeakyEvent&& event_to_leaky_event,
                    HandleLeakyEvent&& handle_leaky_event)
{
  return LeakySampling<Event, EventToLeakyEvent, HandleLeakyEvent>(
      width, height, width_sampling_factor, height_sampling_factor,
      sampling_factor, std::forward<EventToLeakyEvent>(event_to_leaky_event),
      std::forward<HandleLeakyEvent>(handle_leaky_event));
}
}  // namespace ETTCM

#endif  // ETTCM_LEAKY_SAMPLING_HPP
