#pragma once

#include "Vec3.h"

//Ray의 멤버 함수가 커널에서 호출 가능하도록 지정자 사용
//커널에서 픽셀별로 Ray를 만들고 RayColor()로 방향 기반 배경색을 계산

//렌더 커널에서 픽셀마다 생성되며 각 멤버 함수와 at(t)같은 접근/계산 함수들이 디바이스 코드에서 호출됨
class Ray
{
public:
	//커널 내부에서만 쓰도록 설계했기에 __device__로도 충분
	DEVICE Ray() {}

	DEVICE Ray(const Point3& origin, const Vector3& direction) : orig(origin), dir(direction) {}
	DEVICE Ray(const Ray& other) : orig(other.orig), dir(other.dir) {}

	DEVICE const Point3& origin() const { return orig; }
	DEVICE const Vector3& direction() const { return dir; }

	//P(t) = A + tB
	DEVICE Point3 at(double t) const { return orig + t * dir; }

private:
	Point3 orig;
	Vector3 dir;
};