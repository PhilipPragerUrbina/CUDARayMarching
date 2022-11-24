//
// Created by Philip on 11/6/2022.
//

#pragma once
#include <cuda_runtime.h>
#include <ostream>

//ease opengl conversion
#define vec3 Vector3

//defined if being compiled using nvcc rather than nvrtc
#define FULL_COMPILATION

/// Cuda Vector3 class
/// @see Based off of TensorMath
/// @details Works on device and host
class Vector3 {
private:
    /*
     * build in cuda type double3 can be used instead and may provide better performance, since the compiler can make better optimizations
     * I decided not to use this, such that operations can be simplified with loops, by accessing the index of a component rather than using x,y,z
     */
    double m_data[3]; ///< m_data is stored in a double array;

    /// Modulo for doubles, implemented like GLSL
    __device__ __host__ static inline double mod( double x, double y){
        return x - y * floor(x/y);
    }
public:
    /// Create a null vector
    /// All components are 0
    __device__ __host__ Vector3() : m_data{0, 0, 0}{}

    /// Construct a vector from a scalar
    /// Sets all components to this value
    /// @param scalar The scalar value
    __device__ __host__ Vector3(double scalar) : m_data{scalar, scalar, scalar} {}

    /// Create a vector with specific values
    /// @param x index 0
    /// @param y index 1
    /// @param z index 2
    __device__ __host__ Vector3(double x, double y, double z) : m_data{x, y, z}{}

    /// Create a copy of another vector
    /// @param other
    __device__ __host__ Vector3(const Vector3& other) : m_data{other.m_data[0], other.m_data[1], other.m_data[2]}{}

    /// Get a value from the vector
    /// @param i The index of the component
    /// @return the value at that is at that index
    /// @warning Will cause havoc if out of bounds
    __device__ __host__ double inline operator[](int i) const { return m_data[i]; }

    /// Set a value from the vector
    /// @param i The index of the component
    /// @return A reference to the component
    /// @warning Will cause havoc if out of bounds
    __device__ __host__ double &operator[](int i) {return m_data[i];}

    /// Get the x value
    /// @ details Compiles to [0]
    __device__ __host__ double inline x() const{
        return m_data[0];
    }
    /// Get the y value
    /// @ details Compiles to [1]
    __device__ __host__ double inline y() const{
        return m_data[1];
    }
    /// Get the z value
    /// @ details Compiles to [2]
    __device__ __host__ double inline z() const{
        return m_data[2];
    }

    /// Assign one vector to another
    /// @param other The vector to get m_data from
    /// @return Reference to the vector
    __device__ __host__ Vector3& operator= (const Vector3& other){
        if(&other != this){ //check self assignment
            m_data[0] = other.m_data[0];
            m_data[1] = other.m_data[1];
            m_data[2] = other.m_data[2];
        }
        return *this;
    }

    /// Negate a vector
    /// a * -1
    /// @return negated vector
    __device__ __host__ inline Vector3 operator-() const{
        return {-m_data[0], -m_data[1], -m_data[2]};
    }

    /// Add the components of two vectors
    /// a + b
    /// @return A new sum vector
    __device__ __host__ friend Vector3 inline operator+(const Vector3& a, const Vector3& b){
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }
    /// Subtract the components of two vectors
    /// a - b
    /// @return A new vector with the subtracted values
    __device__ __host__ friend Vector3 inline operator-(const Vector3& a, const Vector3& b){
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }
    /// Multiply the components of two vectors
    /// a * b
    /// @return A new product vector
    __device__ __host__ friend Vector3 inline operator*(const Vector3& a, const Vector3& b){
        return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
    }
    /// Divide the components of two vectors
    /// a / b
    /// @return A new vector with the divided values
    __device__ __host__ friend Vector3 inline operator/(const Vector3& a, const Vector3& b){
        return {a[0] / b[0], a[1] / b[1], a[2] / b[2]};
    }

    /// a = a + other
    /// @param other
    __device__ __host__ void inline operator+=(const Vector3& other){
        *this = *this + other;
    }
    /// a = a - other
    /// @param other
    __device__ __host__ void inline operator-=(const Vector3& other){
        *this = *this - other;
    }
    /// a = a * other
    /// @param other
    __device__ __host__ void inline operator*=(const Vector3& other){
        *this = *this * other;
    }
    /// a = a / other
    /// @param other
    __device__ __host__ void inline operator/=(const Vector3& other){
        *this = *this / other;
    }


#ifdef FULL_COMPILATION //Not supported by NVRTC
    /// Print a vector using console out
    /// @details (x,y,z)
    friend std::ostream &operator<<(std::ostream &os, const Vector3 &vector) {
        os << "(" << vector[0] << "," << vector[1] << "," << vector[2] << ")";
        return os;
    }
#endif

    /// Get the absolute value of each of the vector's components
    /// @return the absolute value vector
    __device__ __host__ Vector3 abs() const{
        return {::abs(m_data[0]), ::abs(m_data[1]), ::abs(m_data[2])};
    }

    /// Get the magnitude or length of a vector
    /// @return The length
    __device__ __host__ double length() const {
        return ::sqrt(m_data[0] * m_data[0] + m_data[1] * m_data[1] + m_data[2] * m_data[2]);
    }

    /// Combine two vectors into a single value
    /// @param other
    /// @return The dot product
    __device__ __host__ double dot(const Vector3 &other) const {
        return m_data[0] * other[0] + m_data[1] * other[1] + m_data[2] * other[2];
    }

    /// Get the perpendicular vector to two other vectors
    /// @remember The right-hand rule
    /// @param other
    /// @return The cross product
    __device__ __host__  Vector3 cross(const Vector3 &other) const {
        return {m_data[1] * other[2] - m_data[2] * other[1],
                m_data[2] * other[0] - m_data[0] * other[2],
                m_data[0] * other[1] - m_data[1] * other[0]};
    }

    /// Normalize a vector
    /// @return The unit vector
    __device__ __host__ Vector3 normalized() const {
        return *this / length();
    }

    /// Reflect a vector over a normal
    /// @param n The normalized m_direction normal
    /// @return The reflected vector
    __device__ __host__ Vector3 reflect(const Vector3& n) const{
        return *this - 2*this->dot(n) * n;
    }

    /// Get the distance between two vectors
    /// @param other
    /// @return distance
    __device__ __host__ double distance(const Vector3 &other) const {
        return (*this-other).length();
    }

    /// Modulus each component by the component of another vector
    /// @param value
    /// @return modulus vector
    __device__ __host__ Vector3 mod(Vector3 value){
        return {mod(m_data[0] , value[0]), mod(m_data[1] , value[1]), mod(m_data[2], value[2])};
    }


};


