#include <stdio.h>
#include <stdlib.h>
#include <png.h>

#include "pngio.h"

unsigned char* readImage(char const * file_name, int *width, int *height) {
	unsigned char *pixel;
	// Initialize the image from a PNG file

	// Make sure the file is readable
	FILE *fp = fopen(file_name, "rb");
	if (!fp) {
		printf("Could not open %s\n", file_name);
		exit(1);
	}

	// Make sure the file is a PNG
	png_byte header[8];    // 8 is the maximum size that can be checked
	fread(header, 1, 8, fp);
	if (png_sig_cmp(header, 0, 8)){
		printf("File %s is not a proper PNG file\n", file_name);
		fclose(fp);
		exit(1);
	}

	// PNG information
	png_structp ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info = png_create_info_struct(ptr);
	setjmp(png_jmpbuf(ptr));
	png_init_io(ptr, fp);
	png_set_sig_bytes(ptr, 8);
	png_read_info(ptr, info);

	png_byte color_type = png_get_color_type(ptr, info);
	png_byte bit_depth = png_get_bit_depth(ptr, info);

	*width = png_get_image_width(ptr, info);
	int width_png = *width;
	*height = png_get_image_height(ptr, info);

	int number_of_passes = png_set_interlace_handling(ptr);
	png_read_update_info(ptr, info);

	// Read the image data
	setjmp(png_jmpbuf(ptr));
	png_bytep* row = (png_bytep*) _mm_malloc(sizeof(png_bytep) * *height, 32);
	#pragma vector aligned
	for (int i = 0; i < *height; i++)
		row[i] = (png_byte*) _mm_malloc(png_get_rowbytes(ptr, info), 32);
	png_read_image(ptr, row);

	if(png_get_rowbytes(ptr, info) != width_png) {
		printf("Error: the image is not in grayscale\n");
	}

	fclose(fp);

	// Convert from png_bytep to byte
	pixel = (unsigned char*)_mm_malloc(sizeof(unsigned char)* (*width)* (*height), 32);

	for(int i = 0; i < *height; i++)
		#pragma vector aligned
		for(int j = 0; j < *width; j++)
			pixel[i* (*width)+j] = (unsigned char)row[i][j];

	for (int i = 0; i < *height; i++)
		_mm_free(row[i]);
	_mm_free(row);

	return(pixel);
}

void writeImage(char const * file_name, unsigned char *pixel, int width, int height) {
	// Open file
	FILE *fp = fopen(file_name, "wb");
	if (!fp) {
		printf("Could not open %s for writing\n", file_name);
		exit(1);
	}

	// Create handes
	png_structp ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);;
	png_infop info = png_create_info_struct(ptr);
	setjmp(png_jmpbuf(ptr));

	png_init_io(ptr, fp);

	// Header for 8-bit grayscale
	png_set_IHDR(ptr, info, width, height, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(ptr, info);

	// Contents of the file
	png_bytep row = (png_bytep) _mm_malloc(sizeof(png_byte)*width, 4);
	for(int i = 0; i < height; i++) {
		#pragma vector aligned
		for(int j = 0; j < width; j++) {
			png_byte t = (png_byte)pixel[i*width+j];
			if (t < 0) t = 0;
			if (t > 255) t = 255;
			row[j] = (png_byte) t;
		}
		png_write_row(ptr, row);
	}

	// Deallocate temporary data
	png_write_end(ptr, NULL);
	png_free_data(ptr, info, PNG_FREE_ALL, -1);
	png_destroy_write_struct(&ptr, (png_infopp)NULL);
	fclose(fp);
	_mm_free(row);
}


