/*
 * ExplicitSim - Software for solving PDEs using explicit methods.
 * Copyright (C) 2017  <Konstantinos A. Mountris> <konstantinos.mountris@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Contributors (alphabetically):
 *      George C. BOURANTAS
 *      Grand R. JOLDES
 *      Konstantinos A. MOUNTRIS
 */

/*!
   \file timer.hpp
   \brief Timer class header file.
   \author Konstantinos A. Mountris
   \date 17/05/2017
*/

#ifndef EXPLICITSIM_UTILITIES_TIMER_HPP_
#define EXPLICITSIM_UTILITIES_TIMER_HPP_


#include <iostream>
#include <chrono>
#include <string>

namespace ExplicitSim {

/*!
 *  \addtogroup Utilities
 *  @{
 */


/*!
 * \class Timer
 * \brief Class implemmenting a timer for profiling of ExplicitSim.
 */

class Timer
{
public:

    /*!
     * \brief Timer constructor.
     */
    Timer();


    /*!
     * \brief Timer destructor.
     */
    virtual ~Timer();


    /*!
     * \brief Reset the timer.
     * \return [void]
     */
    void Reset();


    /*!
     * \brief Get the elapsed time in milliseconds.
     * \return [double] The elapsed time in milliseconds.
     */
    double ElapsedMilliSecs() const;


    /*!
     * \brief Get the elapsed time in seconds.
     * \return [double] The elapsed time in seconds.
     */
    double ElapsedSecs() const;


    /*!
     * \brief Get the elapsed time in minutes.
     * \return [double] The elapsed time in minutes.
     */
    double ElapsedMinutes() const;


    /*!
     * \brief Get the elapsed time in hours.
     * \return [double] The elapsed time in hours.
     */
    double ElapsedHours() const;


    /*!
     * \brief Print the elapsed time either in milliseconds, seconds, or minutes in string format.
     *        The result depends on the total elapsed time.
     * \return [std::string] The elapsed time.
     */
    std::string PrintElapsedTime() const;

private:
    typedef std::chrono::high_resolution_clock timer_clock_;                /*!< The timer's clock. */

    typedef std::chrono::duration<double, std::ratio<1> > timer_second_;    /*!< The time in seconds. */

    typedef std::chrono::duration<double, std::milli > timer_millisec_;     /*!< The time in milliseconds. */

    std::chrono::time_point<timer_clock_> beg_;                             /*!< The beginning of time. */
};


/*! @} End of Doxygen Groups*/
} //end of namespace ExplicitSim

#endif //EXPLICITSIM_UTILITIES_EXPLICIT_SIM_TIMER_HPP_
