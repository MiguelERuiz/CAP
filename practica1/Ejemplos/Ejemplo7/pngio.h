#ifndef PNGIO_H
#define PNGIO_H

unsigned char* readImage(char const * file_name, int *width, int *height);
void writeImage(char const * file_name, unsigned char *pixel, int width, int height);

#endif
