#include <stdio.h>
#include <stdint.h>
#include "mem_broker.h"

#define MEM_DEMO_SIZE   1 * 1024 * 1024 // 1MB

int main()
{
    /* Allocate EDMC Memory Buffer */
    void *edmc_vir_addr = MemBroker_GetMemory(MEM_DEMO_SIZE, VMF_ALIGN_TYPE_128_BYTE);

    /* Get Physical Address from Virtual Address */
    void *edmc_phy_addr = MemBroker_GetPhysAddr(edmc_vir_addr);

    printf("\nEDMC buffer virtual addr: %p, physical addr: %p\n\n", edmc_vir_addr, edmc_phy_addr);

    // Doing some memory access in cpu, eq. memcpy()
    /* Flush cache data into physical DRAM */
    MemBroker_CacheCopyBack(edmc_vir_addr, MEM_DEMO_SIZE);

    // Doing some memory access in Hardware, eq. DMA, IE/NPU
    /* Make sure there is no cache with data different from physical DRAM  */
    MemBroker_CacheFlush(edmc_vir_addr, MEM_DEMO_SIZE);

    /* Free EDMC Memory Buffer */
    /* !!! MUST Free the Memory, Or System WILL BE memory leaked until reboot. !!! */
    MemBroker_FreeMemory(edmc_vir_addr);

    return 0;
}
