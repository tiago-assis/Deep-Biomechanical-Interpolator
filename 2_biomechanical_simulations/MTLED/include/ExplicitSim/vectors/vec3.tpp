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


#ifndef EXPLICITSIM_VECTORS_VEC3_TPP_
#define EXPLICITSIM_VECTORS_VEC3_TPP_

#include "ExplicitSim/vectors/vec3.hpp"


namespace ExplicitSim {


template <typename DATATYPE>
Vec3<DATATYPE>::Vec3(DATATYPE x, DATATYPE y, DATATYPE z)
    : x_(x), y_(y), z_(z)
{}


template <typename DATATYPE>
Vec3<DATATYPE>::Vec3(const Vec3<DATATYPE> &vec)
    : x_(vec.x_), y_(vec.y_), z_(vec.z_)
{}


template <typename DATATYPE>
Vec3<DATATYPE>::Vec3(Vec3<DATATYPE> &&vec)
   : x_(vec.x_), y_(vec.y_), z_(vec.z_)
{
   vec.x_ = 0; vec.y_ = 0; vec.z_ = 0;
}


template <typename DATATYPE>
Vec3<DATATYPE>::~Vec3()
{}


template <typename DATATYPE>
void Vec3<DATATYPE>::Set(const DATATYPE &x, const DATATYPE &y, const DATATYPE &z)
{
    this->x_ = x;
    this->y_ = y;
    this->z_ = z;
}


template <typename DATATYPE>
void Vec3<DATATYPE>::Set(const Vec3<DATATYPE> &v)
{
    this->x_ = v.x_;
    this->y_ = v.y_;
    this->z_ = v.z_;
}


template <typename DATATYPE>
void Vec3<DATATYPE>::SetX(const DATATYPE &x)
{
    this->x_ = x;
}


template <typename DATATYPE>
void Vec3<DATATYPE>::SetY(const DATATYPE &y)
{
    this->y_ = y;
}


template <typename DATATYPE>
void Vec3<DATATYPE>::SetZ(const DATATYPE &z)
{
    this->z_ = z;
}


template <typename DATATYPE>
void Vec3<DATATYPE>::Scale(const DATATYPE &a)
{
    this->x_ *= a;
    this->y_ *= a;
    this->z_ *= a;
}


template <typename DATATYPE>
DATATYPE Vec3<DATATYPE>::Dot(const Vec3<DATATYPE> &v) const
{
    return (this->x_*v.x_ + this->y_*v.y_ + this->z_*v.z_);
}


template<typename DATATYPE>
Vec3<DATATYPE> Vec3<DATATYPE>::Cross(const Vec3<DATATYPE> &v)
{
    // The cross product of this vector with v vector.
    Vec3<DATATYPE> cross(this->y_*v.z_ - this->z_*v.y_,
                         this->z_*v.x_ - this->x_*v.z_,
                         this->x_*v.y_ - this->y_*v.x_);
    return cross;
}


template<typename DATATYPE>
Vec3<DATATYPE> Vec3<DATATYPE>::CoeffWiseMul(const Vec3<DATATYPE> &v) const
{
    Vec3<DATATYPE> r(this->x_*v.x_, this->y_*v.y_, this->z_*v.z_);

    return r;
}


template<typename DATATYPE>
Vec3<DATATYPE> Vec3<DATATYPE>::CoeffWiseDiv(const Vec3<DATATYPE> &v) const
{
    Vec3<DATATYPE> r(this->x_/v.x_, this->y_/v.y_, this->z_/v.z_);

    return r;
}

template<typename DATATYPE>
Vec3<DATATYPE> Vec3<DATATYPE>::CoeffWiseAbs() const
{
    Vec3<DATATYPE> r(std::fabs(this->x_), std::fabs(this->y_), std::fabs(this->z_));

    return r;
}


template<typename DATATYPE>
DATATYPE Vec3<DATATYPE>::Length2() const
{
    return (this->x_ * this->x_ +
            this->y_ * this->y_ +
            this->z_ * this->z_ );
}


template<typename DATATYPE>
DATATYPE Vec3<DATATYPE>::Distance2(const Vec3<DATATYPE> &v) const
{
    return (this->x_ - v.x_) * (this->x_ - v.x_) +
           (this->y_ - v.y_) * (this->y_ - v.y_) +
           (this->z_ - v.z_) * (this->z_ - v.z_);
}


template<typename DATATYPE>
DATATYPE Vec3<DATATYPE>::MaxCoeff() const
{
    // Return the maximum coefficient value.
    if (this->x_ > this->y_ && this->x_ > this->z_) { return this->x_; }
    else if (this->y_ > this->x_ && this->y_ > this->z_) { return this->y_; }
    else { return this->z_; }
}


template<typename DATATYPE>
DATATYPE Vec3<DATATYPE>::MinCoeff() const
{
    // Return the minimum coefficient value.
    if (this->x_ < this->y_ && this->x_ < this->z_) { return this->x_; }
    else if (this->y_ < this->x_ && this->y_ < this->z_) { return this->y_; }
    else { return this->z_; }
}


template <typename DATATYPE>
DATATYPE & Vec3<DATATYPE>::operator [] (const size_t i)
{
    switch(i) {
        case 0: return this->x_;
        case 1: return this->y_;
        default: return this->z_;
    }
}


template <typename DATATYPE>
DATATYPE Vec3<DATATYPE>::operator [] (const size_t i) const
{
    switch(i) {
        case 0: return this->x_;
        case 1: return this->y_;
        default: return this->z_;
    }
}


template <typename DATATYPE>
bool Vec3<DATATYPE>::operator == (const Vec3<DATATYPE> &v) const
{
    return (std::fabs(this->x_ - v.x_) < 0.00000001 &&
            std::fabs(this->y_ - v.y_) < 0.00000001 &&
            std::fabs(this->z_ - v.z_) < 0.00000001);
}


template <typename DATATYPE>
bool Vec3<DATATYPE>::operator != (const Vec3<DATATYPE> &v) const
{
    return !(*this == v);
}


template <typename DATATYPE>
bool Vec3<DATATYPE>::operator < (const Vec3<DATATYPE> &v) const
{
    // Counter-clockwise ordering.
    return ( (this->z_ < v.z_) ||
             ((this->z_ == v.z_) && (this->y_ < v.y_)) ||
             ((this->z_ == v.z_) && (this->y_ == v.y_) && (this->x_ < v.x_))
           );
}


template <typename DATATYPE>
bool Vec3<DATATYPE>::operator <= (const Vec3<DATATYPE> &v) const
{
    return ( (*this < v) || (*this == v) );
}


template <typename DATATYPE>
bool Vec3<DATATYPE>::operator > (const Vec3<DATATYPE> &v) const
{
    return !(*this < v);
}


template <typename DATATYPE>
bool Vec3<DATATYPE>::operator >= (const Vec3<DATATYPE> &v) const
{
    return ( (*this > v) || (*this == v) );
}


template <typename DATATYPE>
Vec3<DATATYPE> & Vec3<DATATYPE>::operator = (const Vec3<DATATYPE> &vec)
{
    if (this != &vec) {
        this->x_ = vec.x_;
        this->y_ = vec.y_;
        this->z_ = vec.z_;
    }
    return *this;
}


template <typename DATATYPE>
Vec3<DATATYPE> & Vec3<DATATYPE>::operator = (Vec3<DATATYPE> &&vec)
{
    if (this != &vec) {
        this->x_ = vec.x_;
        this->y_ = vec.y_;
        this->z_ = vec.z_;

        vec.x_ = 0;
        vec.y_ = 0;
        vec.z_ = 0;
    }
    return *this;
}


template <typename DATATYPE>
Vec3<DATATYPE> & Vec3<DATATYPE>::operator += (const Vec3<DATATYPE> &v)
{
    this->x_ += v.x_; this->y_ += v.y_; this->z_ += v.z_;
    return *this;
}


template <typename DATATYPE>
Vec3<DATATYPE> & Vec3<DATATYPE>::operator -= (const Vec3<DATATYPE> &v)
{
    this->x_ -= v.x_; this->y_ -= v.y_; this->z_ -= v.z_;
    return *this;
}


template <typename DATATYPE>
Vec3<DATATYPE> Vec3<DATATYPE>::operator *= (const DATATYPE & scalar)
{
    this->x_ *= scalar; this->y_ *= scalar; this->z_ *= scalar;
    return *this;
}


template <typename DATATYPE>
Vec3<DATATYPE> Vec3<DATATYPE>::operator /= (const DATATYPE & scalar)
{
    this->x_ /= scalar; this->y_ /= scalar; this->z_ /= scalar;
    return *this;
}


}  //end of namespace ExplicitSim


#endif  //EXPLICITSIM_VECTORS_VEC3_TPP_
