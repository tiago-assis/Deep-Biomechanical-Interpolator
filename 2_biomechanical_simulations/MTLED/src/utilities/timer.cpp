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


#include "ExplicitSim/utilities/timer.hpp"


namespace ExplicitSim {

Timer::Timer() : beg_(timer_clock_::now())
{}


Timer::~Timer()
{}


void Timer::Reset()
{
    this->beg_ = timer_clock_::now();
}


double Timer::ElapsedMilliSecs() const
{
    return std::chrono::duration_cast<timer_millisec_>
            (timer_clock_::now() - beg_).count();
}


double Timer::ElapsedSecs() const
{
    return std::chrono::duration_cast<timer_second_>
            (timer_clock_::now() - beg_).count();
}


double Timer::ElapsedMinutes() const
{
    return std::chrono::duration_cast<timer_second_>
            (timer_clock_::now() - beg_).count() / 60.;
}


double Timer::ElapsedHours() const
{
    return std::chrono::duration_cast<timer_second_>
            (timer_clock_::now() - beg_).count() / 3600.;
}


std::string Timer::PrintElapsedTime() const
{
    if (this->ElapsedMilliSecs() < 1000.) { return std::to_string(this->ElapsedMilliSecs()) + " ms"; }
    else if (this->ElapsedMilliSecs() > 1000. &&
             this->ElapsedSecs() < 60.) { return std::to_string(this->ElapsedSecs()) + " s"; }
    else if (this->ElapsedSecs() > 60. &&
             this->ElapsedSecs() < 3600.) { return std::to_string(this->ElapsedMinutes()) + " mins"; }
    else { return std::to_string(this->ElapsedHours()) + " hours"; }
}


}  //end of namespace ExplicitSim
