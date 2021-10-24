#include <math.h>
#include "nbody.h"

void bodyForce(body *p, float dt, int n) {
	float G = 6.674e-11;
	float softeningSquared = 1e-3;

	for (int i = 0; i < n; i++) {
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		#pragma loop count(1000)
		#pragma vector aligned
		for (int j = 0; j < n; j++) {
			float dx = p->x[j] - p->x[i];
			float dy = p->y[j] - p->y[i];
			float dz = p->z[j] - p->z[i];
			float distSqr = dx*dx + dy*dy + dz*dz + softeningSquared;
			float invDist = 1.0f / sqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;
			float g_masses = G * p->m[j] * invDist3;
			Fx += g_masses * dx;
			Fy += g_masses * dy;
			Fz += g_masses * dz;
		}

		p->vx[i] += dt*Fx;
		p->vy[i] += dt*Fy;
		p->vz[i] += dt*Fz;
	}
}

void integrate(body *p, float dt, int n){
	#pragma ivdep
	#pragma vector aligned
	for (int i = 0 ; i < n; i++) {
		p->x[i] += p->vx[i]*dt;
		p->y[i] += p->vy[i]*dt;
		p->z[i] += p->vz[i]*dt;
	}
}
