#pragma once

#include "Types.h"

#include <cmath>
#include <iostream>

//Vector3의 모든 연산이 CPU/GPU 양쪽에서 컴파일되도록 정리
//커널에서 픽셀 값을 float* 3채널 배열이 아닌 Color(Vec3) 객체로 다루도록 함

//커널 직접/간접 호출 될 수 있는 모든 함수에 HOSTDEVICE (__host__ __device__) 를 사용

//std 수학함수는 동작하지 않는 경우가 있어서 CUDA 내장 수학 함수를 사용
struct Vector3 
{
	double e[3];

	HOSTDEVICE Vector3() : e{ 0,0,0 } {}
	HOSTDEVICE Vector3(double e0, double e1, double e2) : e{ e0,e1,e2 } {}

	HOSTDEVICE double x() const { return e[0]; }
	HOSTDEVICE double y() const { return e[1]; }
	HOSTDEVICE double z() const { return e[2]; }

	HOSTDEVICE Vector3 operator-() const { return Vector3(-e[0], -e[1], -e[2]); }
	HOSTDEVICE double operator[](int i) const { return e[i]; }
	HOSTDEVICE double& operator[](int i) { return e[i]; }

	HOSTDEVICE Vector3& operator+=(const Vector3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	HOSTDEVICE Vector3& operator*=(double t) {
		e[0] *= t;
		e[1] *= t;
		e[2] *= t;
		return *this;
	}

	HOSTDEVICE Vector3& operator/=(double t) 
	{
		return *this *= 1 / t;
	}

	HOSTDEVICE double Length() const 
	{
		return sqrt(LengthSquared());
	}

	HOSTDEVICE double LengthSquared() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	HOSTDEVICE bool NearZero() const
	{
		auto threshhold = 1e-8; //1 x 10^-8

		return (fabs(e[0]) < threshhold)
			&& (fabs(e[1]) < threshhold)
			&& (fabs(e[2]) < threshhold);
	}
};

typedef Vector3 Vec3;

using Point3 = Vector3;
using Color = Vector3;

//벡터 유틸리티 함수
inline std::ostream& operator<<(std::ostream& out, const Vector3& v)
{
	return out << v.e[0] 
		<< ' ' << v.e[1] 
		<< ' ' << v.e[2];
}

HOSTDEVICE inline Vector3 operator+(const Vector3& u, const Vector3& v)
{
	return Vector3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

HOSTDEVICE inline Vector3 operator-(const Vector3& u, const Vector3& v)
{
	return Vector3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

HOSTDEVICE inline Vector3 operator*(const Vector3& u, const Vector3& v)
{
	return Vector3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

HOSTDEVICE inline Vector3 operator*(double t, const Vector3& v)
{
	return Vector3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

HOSTDEVICE inline Vector3 operator*(const Vector3& v, double t)
{
	return t * v;
}

HOSTDEVICE inline Vector3 operator/(const Vector3& v, double t)
{
	return (1 / t) * v;
}

HOSTDEVICE inline double Dot(const Vector3& u, const Vector3& v)
{
	return u.e[0] * v.e[0]
		 + u.e[1] * v.e[1]
		 + u.e[2] * v.e[2];
}

HOSTDEVICE inline Vector3 Cross(const Vector3& u, const Vector3& v)
{
	return Vector3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
				   u.e[2] * v.e[0] - u.e[0] * v.e[2],
				   u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

HOSTDEVICE inline Vector3 UnitVector(const Vector3& v)
{
	return v / v.Length();
}

HOSTDEVICE inline Vector3 Reflect(const Vector3& v, const Vector3& n)
{
	return v - 2.0 * Dot(v, n) * n;
}

HOSTDEVICE inline Vector3 Refract(const Vector3& uv, const Vector3& n, double etaInOverEtaOut)
{
	const double cosTheta = fmin(Dot(-uv, n), 1.0);

	const Vector3 refractPerpendicular = etaInOverEtaOut * (uv + cosTheta * n);
	const Vector3 refractParallel = -sqrt(fabs(1.0 - refractPerpendicular.LengthSquared())) * n;

	return refractPerpendicular + refractParallel;
}