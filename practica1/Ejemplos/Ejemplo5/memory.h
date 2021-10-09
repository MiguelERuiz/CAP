#ifndef MEMORY_H
#define MEMORY_H

body* get_memory(int nBodies);
void free_memory(body *p);

void randomizeBodies(body *data, int n);

#endif
