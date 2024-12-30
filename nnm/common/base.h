/**
 * @file      base.h
 * @brief     Basic utils & struct
 * @copyright (c) 2018-2023 Kneron Inc. All right reserved.
 */

#ifndef __BASE_H__
#define __BASE_H__

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define EPSILON_FLT                     ((float)0.000001)   /**< a loose value than FLT_FPSILON */
typedef uint8_t                         u8;
typedef uint16_t                        u16;
typedef uint32_t                        u32;
typedef uint64_t                        u64;

typedef uint8_t                         u8_t;
typedef uint16_t                        u16_t;
typedef uint32_t                        u32_t;
typedef uint64_t                        u64_t;

typedef int8_t                          s8;
typedef int16_t                         s16;
typedef int32_t                         s32;
typedef int64_t                         s64;

#ifndef BIT
#define BIT(x)                          (0x01U << (x))
#endif

#define BIT0                            0x00000001
#define BIT1                            0x00000002
#define BIT2                            0x00000004
#define BIT3                            0x00000008
#define BIT4                            0x00000010
#define BIT5                            0x00000020
#define BIT6                            0x00000040
#define BIT7                            0x00000080
#define BIT8                            0x00000100
#define BIT9                            0x00000200
#define BIT10                           0x00000400
#define BIT11                           0x00000800
#define BIT12                           0x00001000
#define BIT13                           0x00002000
#define BIT14                           0x00004000
#define BIT15                           0x00008000
#define BIT16                           0x00010000
#define BIT17                           0x00020000
#define BIT18                           0x00040000
#define BIT19                           0x00080000
#define BIT20                           0x00100000
#define BIT21                           0x00200000
#define BIT22                           0x00400000
#define BIT23                           0x00800000
#define BIT24                           0x01000000
#define BIT25                           0x02000000
#define BIT26                           0x04000000
#define BIT27                           0x08000000
#define BIT28                           0x10000000
#define BIT29                           0x20000000
#define BIT30                           0x40000000
#define BIT31                           0x80000000

#ifndef NULL
#define NULL                            0
#endif

#ifndef ENABLE
#define ENABLE                          1
#endif

#ifndef DISABLE
#define DISABLE                         0
#endif

#ifndef FALSE
#define FALSE                           0
#endif

#ifndef TRUE
#define TRUE                            1
#endif

#ifndef FAIL
#define FAIL                            0
#endif

#ifndef SUCCESS
#define SUCCESS                         1
#endif

#ifndef boolean
typedef char                            boolean;
#endif

#ifndef MIN
#define MIN(a,b)                        (((a)<(b))?(a):(b))
#endif

#ifndef MAX
#define MAX(a,b)                        (((a)>(b))?(a):(b))
#endif

#ifndef CLIP
#define CLIP(n, lower, upper)           (MAX(lower, MIN(n, upper)))
#endif

#ifndef ABSDIFF
#define ABSDIFF(x, y)                   (x > y) ? (x-y) : (y-x)
#endif

#ifndef ABS
#define ABS(a)                          (((a)>=0)?(a):(-(a)))
#endif

#ifndef FLOOR
#define FLOOR(val)                      ((int)(val) - ((int)(val) > val))
#endif

#ifndef ROUND
#define ROUND(x)                        ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))
#endif

#define STS_OK                          0
#define STS_ERR_NORMAL                  1
#define STS_ERR_CRC                     2

#endif /* __BASE_H_ */
