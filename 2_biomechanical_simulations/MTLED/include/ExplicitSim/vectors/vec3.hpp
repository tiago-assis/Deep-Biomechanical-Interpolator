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
   \file vec3.hpp
   \brief Vec3 class header file.
   \author Konstantinos A. Mountris
   \date 28/02/2017
*/


#ifndef EXPLICITSIM_VECTORS_VEC3_HPP_
#define EXPLICITSIM_VECTORS_VEC3_HPP_


#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>

#include <vector>
#include <set>
#include <cstring>
#include <string>

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>


namespace ExplicitSim {

/*!
 *  \addtogroup Vectors
 *  @{
 */

/*!
 * \class Vec3
 * \brief A templated class implenting a 3D vector.
 */

template <typename DATATYPE>
class Vec3{
public:

    /*!
     * \brief Vec3 constructor.
     *
     * Vec3 constructor initializes a Vec3 object with default zero coordinates,
     *      except if given differently by the user.
     *
     * \param [in] x X coordinate.
     * \param [in] y Y coordinate.
     * \param [in] z Z coordinate.
     */
    Vec3(DATATYPE x = 0, DATATYPE y = 0, DATATYPE z = 0);


    /*!
     * \overload
     * \brief Vec3 copy consturctor.
     * \param [in] vec The Vec3 vector to be copied.
     */
    Vec3(const Vec3<DATATYPE> &vec);


    /*!
     * \overload
     * \brief Vec3 move consturctor.
     * \param [in] vec The Vec3 vector to be moved.
     */
    Vec3(Vec3<DATATYPE> &&vec);


    /*!
     * \brief Vec3 vector destructor.
     */
    virtual ~Vec3();


    /*!
     * \brief Set Vec3 vector coordinates.
     *
     * Three values of @type DATATYPE  given by the user are
     * associated to the coordinates of the vector.
     *
     * \param [in] x X coordinate.
     * \param [in] y Y coordinate.
     * \param [in] z Z coordinate.
     * \return Void.
     */
    void Set(const DATATYPE &x, const DATATYPE &y, const DATATYPE &z);


    /*!
     * \brief Set Vec3 vector coordinates by copying
     *        coordinates of another Vec3 vector.
     * \param [in] v Vec3 vector to copy coordinates from.
     * \return [void].
     */
    void Set(const Vec3<DATATYPE> &v);


    /*!
     * \brief Set Vec3 vector X coordinate.
     * \param [in] x The X coordinate.
     * \return [void].
     */
    void SetX(const DATATYPE &x);


    /*!
     * \brief Set Vec3 vector Y coordinate.
     * \param [in] y The Y coordinate.
     * \return [void].
     */
    void SetY(const DATATYPE &y);


    /*!
     * \brief Set Vec3 vector Z coordinate.
     * \param [in] z The Z coordinate.
     * \return [void].
     */
    void SetZ(const DATATYPE &z);


    /*!
       \brief Scales the Vec3 vector by a given value.
       \param [in] a The Scaling value.
       \return [void].
     */
    void Scale(const DATATYPE &a);


    /*!
       \brief Calculates the scalar product with another Vec3 vector.
       \param [in] v A Vec3 vector.
       \return [DATATYPE] The scalar product value of the two Vec3 vectors.
    */
    DATATYPE Dot(const Vec3<DATATYPE> &v) const;


    /*!
     * \brief Camculates the Vec3 vector cross product.
     * @param [in] v Vec3 vector to apply cross product.
     */
    Vec3<DATATYPE> Cross(const Vec3<DATATYPE> &v);


    /*!
     * \brief Multiply the coefficients of this vector with the coefficient of vector vec.
     * \param [in] v The vector to be used for coefficient-wise multiplication.
     * \return [ExplicitSim::Vec3<DATATYPE>] The coefficient-wise multiplication product.
     */
    Vec3<DATATYPE> CoeffWiseMul(const Vec3<DATATYPE> &v) const;


    /*!
     * \brief Divide the coefficients of this vector with the coefficient of vector vec.
     * \param [in] v The vector to be used for coefficient-wise division.
     * \return [ExplicitSim::Vec3<DATATYPE>] The coefficient-wise division product.
     */
    Vec3<DATATYPE> CoeffWiseDiv(const Vec3<DATATYPE> &v) const;


    /*!
     * \brief Convert the coefficients of this vector to absolute values.
     * \return [ExplicitSim::Vec3<DATATYPE>] This vector with absolute values at its coefficients.
     */
    Vec3<DATATYPE> CoeffWiseAbs() const;


    /*!
     * \brief Calculate the squared length of a 3D vector.
     */
    DATATYPE Length2() const;

    /*!
     * \brief Calculate the squared distance between two 3D vectors.
     */
    DATATYPE Distance2(const Vec3<DATATYPE> &v) const;


    /*!
     * \brief Get the value of the maximum coefficient of the 3D vector.
     * \return [DATATYPE] The value of the maximum coefficient of the 3D vector.
     */
    DATATYPE MaxCoeff() const;


    /*!
     * \brief Get the value of the minimum coefficient of the 3D vector.
     * \return [DATATYPE] The value of the minimum coefficient of the 3D vector.
     */
    DATATYPE MinCoeff() const;



    /*!
     * Accesses with writing permission the i-th coordinate of the Vec3 vector.
     *
     * \param [in] i The i-th coordinate of the Vec3 vector.
     * \return [DATATYPE] The i-th coordinate of the Vec3 vector.
     */
    DATATYPE & operator [] (const size_t i);


    /*!
     * Read only access to the i-th coordinate of the Vec3 vector.
     *
     * \param [in] i The i-th coordinate of the Vec3 vector.
     * \return [DATATYPE] The i-th coordinate of the Vec3 vector.
     */
    DATATYPE operator [] (const size_t i) const;


    /*!
     * \param [in] v The Vec3 vector to be compared.
     * \return [bool] True if Vec3 vectors are equal.
     */
    bool operator == (const Vec3<DATATYPE> &v) const;


    /*!
       \param [in] v The Vec3 vector to be compared.
       \return [bool] True if Vec3 vectors are not equal.
    */
    bool operator != (const Vec3<DATATYPE> &v) const;


    /*!
       The Vec3 vectors are compared in the X coordinate. If X coordinates
       of the two Vec3 vectors are equal then Y coordinates are compared.
       If Y coordinates are also equal then comparison of the Z coordinates
       is performed.

       \param [in] v The Vec3 vector to be compared.
       \return [bool] True if Vec3 vector is  smaller than v.
    */
    bool operator < (const Vec3<DATATYPE> &v) const;


    /*!
       \param [in] v The Vec3 vector to be compared.
       \return [bool] True if Vec3 vector is  smaller than or equal with v.
    */
    bool operator <= (const Vec3<DATATYPE> &v) const;


    /*!
       The Vec3 vectors are compared in the X coordinate. If X coordinates
       of the two Vec3 vectors are equal then Y coordinates are compared.
       If Y coordinates are also equal then comparison of the Z coordinates
       is performed.

       \param [in] v The Vec3 vector to be compared.
       \return [bool] True if Vec3 vector is  bigger than v.
    */
    bool operator > (const Vec3<DATATYPE> &v) const;


    /*!
       \param [in] v The Vec3 vector to be compared.
       \return [bool] True if Vec3 vector is  bigger than or equal with v.
    */
    bool operator >= (const Vec3<DATATYPE> &v) const;


    /*!
     * Vec3 copy assignment operator.
     * \param [in] vec The vector to be copy-assigned.
     * \return [ExplicitSim::Vec3<DATATYPE>] The Vec3 vector containing the data of vec.
     */
    Vec3<DATATYPE> & operator = (const Vec3<DATATYPE> &vec);


    /*!
     * Vec3 move assignment operator.
     * \param [in] vec The vector to be move-assigned.
     * \return [ExplicitSim::Vec3<DATATYPE>] The Vec3 vector containing the data of vec.
     */
    Vec3<DATATYPE> & operator = (Vec3<DATATYPE> &&vec);


    /*!
       \param [in] v The Vec3 vector to be added in place.
       \return [ExplicitSim::Vec3<DATATYPE>] The Vec3 vector containing the result from the additive assignment.
    */
    Vec3<DATATYPE> & operator += (const Vec3<DATATYPE> &v);


    /*!
     * \param [in] v The Vec3 vector to be subtracted in place.
     * \return [ExplicitSim::Vec3<DATATYPE>] The Vec3 vector containing the result from the subtracting assignment.
     */
    Vec3<DATATYPE> & operator -= (const Vec3<DATATYPE> &v);


    /*!
     * \brief Operator for coefficient-wise multiplication assignment with a scalar.
     * \param [in] val The value to multiply the vector coefficients.
     * \return [ExplicitSim::Vec3<DATATYPE>] The resulting Vec3 vector from the subtraction.
     */
    Vec3<DATATYPE> operator *= (const DATATYPE & scalar);


    /*!
     * \brief Operator for coefficient-wise division assignment with a scalar.
     * \param [in] val The value to divide the vector coefficients.
     * \return [ExplicitSim::Vec3<DATATYPE>] The resulting Vec3 vector from the division.
     */
    Vec3<DATATYPE> operator /= (const DATATYPE &scalar);


    /*!
       \brief Read-only access to the X coordinate of the Vec3 vector.
       \return The X coordinate of the Vec3 vector.
    */
    inline const DATATYPE & X() const { return this->x_; }


    /*!
       \brief Read-only access to the Y coordinate of the Vec3 vector.
       \return The Y coordinate of the Vec3 vector.
    */
    inline const DATATYPE & Y() const { return this->y_; }


    /*!
       \brief Read-only access to the Z coordinate of the Vec3 vector.
       \return The Z coordinate of the Vec3 vector.
    */
    inline const DATATYPE & Z() const { return this->z_; }


    /*!
       \brief Stream output of the Vec3 vector.
       \param [out] out Stream for output.
       \param [in] v The Vec3 vector to be flushed in the output.
       \return Output the coordinates of Vec3 vector v in std::string format.
    */
    inline friend std::ostream & operator << (std::ostream &out, const Vec3<DATATYPE> &v)
    {
        out << v.x_ << " " << v.y_ << " " << v.z_;
        return out;
    }


private:
    DATATYPE x_;        /*!< X coordinate of the Vec3 vector. */
    DATATYPE y_;        /*!< Y coordinate of the Vec3 vector. */
    DATATYPE z_;        /*!< Z coordinate of the Vec3 vector. */

};


template<typename DATATYPE>
Vec3<DATATYPE> operator + (Vec3<DATATYPE> vec1, const Vec3<DATATYPE> &vec2)
{
    return vec1 += vec2;
}

template<typename DATATYPE>
Vec3<DATATYPE> operator - (Vec3<DATATYPE> vec1, const Vec3<DATATYPE> &vec2)
{
    return vec1 -= vec2;
}


template<typename DATATYPE>
Vec3<DATATYPE> operator * (Vec3<DATATYPE> vec, const DATATYPE &scalar)
{
    return vec *= scalar;
}


template<typename DATATYPE>
Vec3<DATATYPE> operator * (const DATATYPE &scalar, Vec3<DATATYPE> vec)
{
    return vec *= scalar;
}


template <typename DATATYPE>
Vec3<DATATYPE> operator / (Vec3<DATATYPE> vec, const DATATYPE &scalar)
{
    return vec /= scalar;
}


/*! @} End of Doxygen Groups*/


} // End of namespace ExplicitSim


#include "ExplicitSim/vectors/vec3.tpp"

#endif //EXPLICITSIM_VECTORS_VEC3_HPP_
