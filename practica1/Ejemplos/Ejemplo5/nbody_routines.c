#include <math.h>
#include "nbody.h"

void bodyForce(body *p, float dt, int n) {

	for (int i = 0; i < n; i++) {
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		#pragma loop count(1000)
		for (int j = 0; j < n; j++) {
			if (i!=j) {
				float dx = p->x[j] - p->x[i];
				float dy = p->y[j] - p->y[i];
				float dz = p->z[j] - p->z[i];
				double softeningSquared = 1e-3;
				float distSqr = dx*dx + dy*dy + dz*dz + softeningSquared;
				float invDist = 1.0f / sqrtf(distSqr);
				float invDist3 = invDist * invDist * invDist;

				double G = 6.674e-11;
				float g_masses = G * p->m[j] * p->m[i];

				Fx += g_masses * dx * invDist3;
				Fy += g_masses * dy * invDist3;
				Fz += g_masses * dz * invDist3;
			}
		}

		p->vx[i] += dt*Fx/p->m[i];
		p->vy[i] += dt*Fy/p->m[i];
		p->vz[i] += dt*Fz/p->m[i];
	}
}

void integrate(body *p, float dt, int n){
	#pragma ivdep
	for (int i = 0 ; i < n; i++) {
		p->x[i] += p->vx[i]*dt;
		p->y[i] += p->vy[i]*dt;
		p->z[i] += p->vz[i]*dt;
	}
}
