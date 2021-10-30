#include "stencil.h"
#include <immintrin.h>

void ApplyStencil(unsigned char *img_in, unsigned char *img_out, int width, int height) {

	__assume_aligned(img_in, 16);
	__assume_aligned(img_out, 16);
	__m256i val;

	for (int i = 1; i < height-1; i++)
		#pragma vector nontemporal (img_in, img_out)
		for (int j = 1; j < width-1; j++) {
			__m256i center = _mm256_load_si256((__m256i*)&img_in[(i)*width + j]);
			__m256i px1 = _mm256_load_si256((__m256i*)&img_in[(i-1)*width + j-1]);
			__m256i px2 = _mm256_load_si256((__m256i*)&img_in[(i-1)*width + j]);
			__m256i px3 = _mm256_load_si256((__m256i*)&img_in[(i-1)*width + j+1]);
			__m256i px4 = _mm256_load_si256((__m256i*)&img_in[(i)*width + j-1]);
			__m256i px5 = _mm256_load_si256((__m256i*)&img_in[(i)*width + j+1]);
			__m256i px6 = _mm256_load_si256((__m256i*)&img_in[(i+1)*width + j-1]);
			__m256i px7 = _mm256_load_si256((__m256i*)&img_in[(i+1)*width + j]);
			__m256i px8 = _mm256_load_si256((__m256i*)&img_in[(i+1)*width + j+1]);

			val = center;
			val = _mm256_subs_epu8(val, px1);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px2);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px3);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px4);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px5);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px6);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px7);
			val = _mm256_adds_epu8(val, center);
			val = _mm256_subs_epu8(val, px8);

			_mm256_store_si256((__m256i*)&img_out[i*width + j], val);
		}
}

void CopyImage(unsigned char *img_in, unsigned char *img_out, int width, int height) {
	__assume_aligned(img_in, 16);
	__assume_aligned(img_out, 16);
	for (int i = 0; i < height; i++)
		#pragma vector aligned
		for (int j = 0; j < width; j++)
			img_in[i*width + j] = img_out[i*width + j];
}
