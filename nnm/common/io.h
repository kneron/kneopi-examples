/**
 * @file      io.h
 * @brief     Basic io access
 * @copyright (c) 2018-2023 Kneron Inc. All right reserved.
 */

#ifndef IO_H
#define IO_H

#include <stdint.h>

#if PRINT_NPU_REG == 1

uint32_t    readl(uint32_t addr);
void        writel(uint32_t val, uint32_t addr);

uint16_t    readw(uint32_t addr);
void        writew(uint16_t val, uint32_t addr);

uint8_t     readb(uint32_t addr);
void        writeb(uint8_t val, uint32_t addr);

uint32_t    ioread32(uint32_t p);
void        iowrite32(uint32_t v, uint32_t p);

#else

#define readl(addr)                                     (*(volatile unsigned int *)(addr))
#define writel(val, addr)                               (*(volatile unsigned int *)(addr) = (val))

#define readw(addr)                                     (*(volatile unsigned short *)(addr))
#define writew(val, addr)                               (*(volatile unsigned short *)(addr) = (val))

#define readb(addr)                                     (*(volatile unsigned char *)(addr))
#define writeb(val, addr)                               (*(volatile unsigned char *)(addr) = (val))

#define ioread32(p)                                     (*(volatile unsigned int *)(p))
#define iowrite32(v, p)                                 (*(volatile unsigned int *)(p) = (v))

#endif    /* PRINT_NPU_REG */

#define inl(p)                                          ioread32(p)
#define outl(v, p)                                      iowrite32(v, p)

#define inw(port)                                       readl(port)
#define outw(port, val)                                 writel(val, port)

#define inb(port)                                       readb(port)
#define outb(port, val)                                 writeb(val, port)

#if defined(KL520)
#define inhw(port)                                      readw(port)
#define outhw(port, val)                                writew(val, port)
#else
#define inhw(port)                                      readl(port)
#define outhw(port, val)                                writel(val, port)
#endif

#if defined(KL520)
#define u32Lib_LeRead32(x)                              *((volatile uint32_t *)((uint8_t * )x))     //bessel:add  (uint8_t * )
#define vLib_LeWrite32(x,y)                             *(volatile uint32_t *)((uint8_t * )x)=(y)   //bessel:add  (uint8_t * )
#else
#define u32Lib_LeRead32(x)                              *((volatile INT32U *)((INT8U * )x))         //bessel:add  (INT8U * )
#define vLib_LeWrite32(x,y)                             *(volatile INT32U *)((INT8U * )x)=(y)       //bessel:add  (INT8U * )
#endif

#define masked_outw(port, val, mask)                    outw(port, (inw(port) & ~mask) | (val & mask))
#define GET_BIT(port, __bit)                            ((inw(port) & BIT##__bit) >> __bit)
#define GET_BITS(port, __s_bit, __e_bit)                ((inw(port) & (BIT##__e_bit | (BIT##__e_bit - BIT##__s_bit))) >> __s_bit)
#define SET_BIT(port, __bit)                            outw(port, BIT##__bit)
#define SET_MASKED_BIT(port, val, __bit)                outw(port, (inw(port) & ~BIT##__bit) | ((val << __bit) & BIT##__bit))
#define SET_MASKED_BITS(port, val, __s_bit, __e_bit)    outw(port, ((inw(port) & ~(BIT##__e_bit | (BIT##__e_bit - BIT##__s_bit))) | (val << __s_bit)));


#endif // IO_H
