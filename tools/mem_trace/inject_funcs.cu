/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint64_t addr,
                                                       int size,
                                                       int offset,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;
    
    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.config.addrs[i] = __shfl_sync(active_mask, addr, i);
    }

    ma.pc = offset;
    unsigned int _smid = get_smid();
    int4 cta = get_ctaid();
    ma.smid = _smid;
    ma.grid_launch_id = grid_launch_id;
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;

    //sg05060
    ma.size = size;

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        // for (int i = 0; i < 32; i++) {
        //     if(ma.config.addrs[i] != 0) {
        //         if(size == 1)
        //             ma.config.data[i] = __ldg((const char*)ma.config.addrs[i]);
        //         else if(size == 4)
        //             ma.config.data[i] = __ldg((const int*)ma.config.addrs[i]);
        //         else if(size == 8)
        //             ma.config.data[i] = __ldg((const double*)ma.config.addrs[i]);
        //     }
        // }
        for (int i = 0; i < 32; i++) {
            if(ma.config.addrs[i] != 0) {
                if(size == 1)
                    ma.config.data_8[i] = __ldg(((const char*)ma.config.addrs[i]));
                else if(size == 2)
                    ma.config.data_16[i] = __ldg(((const char*)ma.config.addrs[i]));
                else if(size == 4)
                    ma.config.data_32[i] = __ldg((const int*)ma.config.addrs[i]);
                else if(size == 8)
                    ma.config.data_64[i] = __ldg((const double*)ma.config.addrs[i]);
                else if(size == 16) {
                    uint8_t* dataPointer = reinterpret_cast<uint8_t*>(ma.config.addrs[i]);
                    for (int j = 0; j < 16; ++j) {
                        ma.config.data_128[i].bytes[j] = __ldg((const char*)(dataPointer)+j);
                    }
                }
            }
        }
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}
