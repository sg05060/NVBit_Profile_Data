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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

/* Compression Algorithm*/
#include "LZW.hpp"

#include <iomanip>
#include <cstdint>
#include <bitset>

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec

#define HEX_DATA(x)                                                  \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << x      \
         << std::dec

#define HEX_DOUBLE(x)                                                  \
    ("0x" + std::to_string(*reinterpret_cast<uint64_t*>(&x)))

#define CHANNEL_SIZE (1l << 20)

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev* channel_dev;
    ChannelHost channel_host;
};

/* lock */
pthread_mutex_t mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

/* Map for [Malloc address : Size] */
std::map<const unsigned long long*, size_t> alloc_address_map;
std::map<unsigned long long, size_t> free_address_map;
std::map<unsigned long long, size_t> memcpy_address_map;
void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", instr_begin_interval,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", instr_end_interval,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", verbose, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* sg05060
    * CUcontext(ctx) : 해당 작업의 GPU 리소스를 관리하게 됨
    * CUfunction(func) : CUcontext내에서 CUDA커널 함수를 실행하며, 커널이 해당 GPU에서 병렬로 
    * 실행되고 결과가 반환된다.
    */ 
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* get vector of instructions of function "f" */
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);

        if (verbose) {
            printf(
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
                instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
                instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT ||
                instr->getMemorySpace() == InstrType::MemorySpace::SURFACE ||
                instr->getMemorySpace() == InstrType::MemorySpace::TEXTURE ||
                instr->getMemorySpace() == InstrType::MemorySpace::SHARED ||
                //instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL_TO_SHARED ||
                instr->getMemorySpace() == InstrType::MemorySpace::GENERIC
                ) {
                cnt++;
                continue;
            }
            if (verbose) {
                instr->printDecoded();
            }

            if (opcode_to_id_map.find(instr->getOpcode()) ==
                opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
            }

            int opcode_id = opcode_to_id_map[instr->getOpcode()];
            int mref_idx = 0;
            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const InstrType::operand_t* op = instr->getOperand(i);

                if (op->type == InstrType::OperandType::MREF) {
                    /* insert call to the instrumentation function with its
                     * arguments */
                    nvbit_insert_call(instr, "instrument_mem", IPOINT_AFTER);
                    /* predicate value */
                    nvbit_add_call_arg_guard_pred_val(instr);
                    /* opcode id */
                    nvbit_add_call_arg_const_val32(instr, opcode_id);
                    /* memory reference 64 bit address */
                    nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                    /* sg05060 */
                    nvbit_add_call_arg_const_val32(instr, op->nbytes);
                    /* add "space" for kernel function pointer that will be set
                     * at launch time (64 bit value at offset 0 of the dynamic
                     * arguments)*/
                    nvbit_add_call_arg_launch_val64(instr, 0);
                    /* add pointer to channel_dev*/
                    nvbit_add_call_arg_const_val64(
                        instr, (uint64_t)ctx_state->channel_dev);
                    mref_idx++;
                }
            }
            cnt++;
        }
    }
}

__global__ void flush_channel(ChannelDev* ch_dev) {
    /* set a CTA id = -1 to indicate communication thread that this is the
     * termination flag */
    mem_access_t ma;
    ma.cta_id_x = -1;
    ch_dev->push(&ma, sizeof(mem_access_t));
    /* flush channel */
    ch_dev->flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
    pthread_mutex_lock(&mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&mutex);
        return;
    }
    skip_callback_flag = true;

    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {
            printf("CTX 0x%016lx - API Name %d\n", (uint64_t)ctx, cbid);
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;

        /* Make sure GPU is idle */
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        } else {
            printf("No CUDA error.\n");
        }
        assert(cudaGetLastError() == cudaSuccess);

        if (!is_exit) {
            /* instrument */
            instrument_function_if_needed(ctx, p->f);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

            /* get function name and pc */
            const char* func_name = nvbit_get_func_name(ctx, p->f);
            uint64_t pc = nvbit_get_func_addr(p->f);

            /* set grid launch id at launch time */
            nvbit_set_at_launch(ctx, p->f, &grid_launch_id, sizeof(uint64_t));
            /* increment grid launch id for next launch */
            grid_launch_id++;

            /* enable instrumented code to run */
            
            nvbit_enable_instrumented(ctx, p->f, false);

            printf(
                "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - Kernel "
                "name %s - grid launch id %ld - grid size %d,%d,%d - block "
                "size %d,%d,%d - nregs %d - shmem %d - cuda stream id %ld\n",
                (uint64_t)ctx, pc, func_name, grid_launch_id, p->gridDimX,
                p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes,
                (uint64_t)p->hStream);
                
            for(auto &pair : alloc_address_map) {
            printf("alloc address : %llu, size : %d\n", *pair.first, pair.second);
            }
            /*
            for(auto &pair : free_address_map) {
                printf("free address : %llu\n", (pair.first));
            }
            for(auto &pair : memcpy_address_map) {
                printf("memcpy address : %llu, size : %d\n", pair.first,pair.second);
            }
            */
                
            double sum = 0;
            size_t count = 0;
        }
    }
            //printf("Map Size : %d\n", mem_address_map.size());
            /*
            for(auto iter: mem_address_map) {
                    char* h_m = (char*)malloc(4096);
                    printf("Address : %lld , Size : %d ", (iter.first), 4096);
                    CUDA_SAFECALL(cudaMemcpy(h_m, (void*)(iter.first), 4096, cudaMemcpyDeviceToHost));
                    printf("MemCpyDtoH complete!\n");

                    std::string hex_str;
                    for(size_t i = 0; i < iter.second; i++) {
                        std::bitset<8> binaryRepresentation(h_m[i]);
                        std::bitset<8> descending_order;
                        std::bitset<4> l_half_byte, r_half_byte;
                        for (int i = 0; i < 4; i++) {
                            l_half_byte[3-i] = binaryRepresentation[i];
                            r_half_byte[3-i] = binaryRepresentation[4+i];
                        }
                        
                        std::stringstream ss;
                        ss << std::hex << l_half_byte.to_ulong() << std::hex << r_half_byte.to_ulong();
                        hex_str += ss.str();
                    }
                    printf("Debug\n");
                    //std::cout << hex_str << endl;

                    LZW compressor(8,9);
                    //compressor.init_table();
                    //std::list<unsigned int> compressed = compressor.compress(hex_str);
                    //printf("CPR : %f\n", compressor.getCPR());
                    //sum += compressor.getCPR();
                    CUDA_SAFECALL(cudaMemcpy((void*)(iter.first), h_m, 4096, cudaMemcpyHostToDevice));
                    free(h_m);
                
            }
            */
            //printf("Average CPR : %f , Page Count : %d\n", (sum/mem_address_map.size()),mem_address_map.size());
    
    /*else if(cbid == API_CUDA_cuMemcpyHtoD ||
        cbid == API_CUDA_cuMemcpyHtoD_v2 ||
        cbid == API_CUDA_cuMemcpyHtoDAsync ||
        cbid == API_CUDA_cuMemcpyHtoDAsync_v2 ||
        cbid == API_CUDA_cuMemcpyHtoD_v2_ptds ||
        cbid == API_CUDA_cuMemcpyHtoDAsync_v2_ptsz) {
        
        cuMemcpyHtoD_v2_params* p_0 = NULL;
        cuMemcpyHtoDAsync_v2_params* p_1 = NULL;
        cuMemAllocManaged_params* p_2 = NULL;
        unsigned long long _dptr;
        unsigned long long mask = ~((1ULL << 9) - 1);
        size_t _bytesize;

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        } else {
            printf("No CUDA error.\n");
        }
        assert(cudaGetLastError() == cudaSuccess);

        if (!is_exit) {
            size_t flag = 0;
            switch (cbid)
            {
            case API_CUDA_cuMemcpyHtoD:
            case API_CUDA_cuMemcpyHtoD_v2:
                p_0 = (cuMemcpyHtoD_v2_params*)params;
                _dptr = (unsigned long long)p_0->dstDevice;
                //printf("memcpy VPN : %lld, Size : %d\n", _dptr, p_0->ByteCount);
                //_dptr &= mask;
                _bytesize = p_0->ByteCount;
                break;
            case API_CUDA_cuMemcpyHtoDAsync:
            case API_CUDA_cuMemcpyHtoDAsync_v2:
            case API_CUDA_cuMemcpyHtoDAsync_v2_ptsz:
                p_1 = (cuMemcpyHtoDAsync_v2_params*)params;
                _dptr = (unsigned long long)p_1->dstDevice;
                //_dptr &= mask;
                _bytesize = p_1->ByteCount;
                //flag = 1;
                break;
            //case API_CUDA_cuMemAllocManaged:
            //    p_2 = (cuMemAllocManaged_params*)params;
            //    _dptr = (unsigned long long*)p_2->dptr;
            //    _bytesize = p_2->bytesize;
            //    break;
            default:
                break;
            }

            
            auto it = memcpy_address_map.find(_dptr);
            if(it != memcpy_address_map.end()) {
                    it->second = _bytesize;
            } else {
                memcpy_address_map.insert({_dptr,_bytesize});
            }
            
            
            printf("CTX 0x%016lx - API Name %d\n", (uint64_t)ctx, cbid);
        }
        
    } 
    */
    else if(cbid == API_CUDA_cuMemFree ||
        cbid == API_CUDA_cuMemFree_v2 ||
        cbid == API_CUDA_cuMemFreeAsync ||
        cbid == API_CUDA_cuMemFreeAsync_ptsz ) {

        cuMemFree_v2_params* p_0 = NULL;
        unsigned long long _dptr;
        unsigned long long mask = ~((1ULL << 9) - 1);
        size_t _bytesize;

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        } else {
            printf("No CUDA error.\n");
        }
        assert(cudaGetLastError() == cudaSuccess);

        if (!is_exit) {
            size_t flag = 0;
            p_0 = (cuMemFree_v2_params*)params;
            _dptr = (unsigned long long)p_0->dptr;
            //_dptr &= mask;
            /*
            auto it = mem_address_map.find(_dptr);
            if(it != mem_address_map.end()) {
                mem_address_map.erase(_dptr);
            } else {

            }
            */
            auto it = free_address_map.find(_dptr);
            if(it == free_address_map.end()) {
                free_address_map.insert({_dptr,0});
            }
            /*
            for(auto pair : alloc_map) {
                printf("alloc map : %ld, size : %d free : %ld\n", *pair.first,pair.second,_dptr);
                if( (*(pair.first)) == _dptr) {
                    for(auto mem_address_pair : mem_address_map) {
                        if((mem_address_pair.first >= _dptr) && (mem_address_pair.first < _dptr + pair.second))
                            mem_address_map.erase(mem_address_pair.first);
                    }
                    alloc_map.erase(pair.first);
                } 
            }
            */
            //printf("Not Found address error\n, VPN : %lld", _dptr);
            printf("CTX 0x%016lx - API Name %d\n", (uint64_t)ctx, cbid);
            //printf("after memfree map size : %d\n", mem_address_map.size());
        }
    } 
    
    else if(cbid == API_CUDA_cuMemAlloc ||
        cbid == API_CUDA_cuMemAlloc_v2 ||
        cbid == API_CUDA_cuMemAllocAsync ||
        cbid == API_CUDA_cuMemAllocAsync_ptsz
    ) { // Page Address Collecting
        cuMemAlloc_params* p_0 = NULL;
        cuMemAllocAsync_params* p_1 = NULL;
        unsigned long long* _dptr;
        //unsigned long long mask = ~((1ULL << 9) - 1);
        unsigned int _bytesize;

        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
        } else {
            printf("No CUDA error.\n");
        }
        assert(cudaGetLastError() == cudaSuccess);

        if (is_exit) {
            switch (cbid)
            {
            case API_CUDA_cuMemAlloc:
            case API_CUDA_cuMemAlloc_v2:
                p_0 = (cuMemAlloc_params*)params;
                _dptr = (unsigned long long*)((p_0->dptr));
                //_dptr &= mask;
                //printf("alloc pointer VPN %lld , Size : %d\n", _dptr,p_0->bytesize);
                //printf("alloc refer VPN : %lld, Size : %d", *(_dptr), p_0->bytesize);
                _bytesize = p_0->bytesize;
                break;
            case API_CUDA_cuMemAllocAsync:
            case API_CUDA_cuMemAllocAsync_ptsz:
                p_1 = (cuMemAllocAsync_params*)params;
                _dptr = (unsigned long long*)((p_1->dptr));
                //_dptr &= mask;
                _bytesize = p_1->bytesize;
                break;
            default:
                break;
            }
            /*
            size_t pcount = (_bytesize % 4096 == 0) ? (_bytesize / 4096) : (_bytesize / 4096) + 1;
            unsigned long long ptr = _dptr;
            for(size_t i = 0; i < pcount; i++) {
                auto it = mem_address_map.find(ptr);
                if(it != mem_address_map.end()) {
                    it->second = _bytesize;
                } else {
                    mem_address_map.insert({ptr,_bytesize});
                }
                ptr+=512;
            }
            */
            
            auto it = alloc_address_map.find((unsigned long long*)(p_0->dptr));
            if(it != alloc_address_map.end()) {
                printf("Find same alloc address...\n");
                it->second = _bytesize;
            } else {
                alloc_address_map.insert({(unsigned long long*)(p_0->dptr),_bytesize});
            }
            //printf("CTX 0x%016lx - API Name %d\n", (uint64_t)ctx, cbid);
            
            printf("CTX 0x%016lx - API Name %d\n", (uint64_t)ctx, cbid);
        }
    } else {
        printf("CTX 0x%016lx - API Name %d\n", (uint64_t)ctx, cbid);
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    bool done = false;
    while (!done) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                mem_access_t* ma =
                    (mem_access_t*)&recv_buffer[num_processed_bytes];

                /* when we receive a CTA_id_x it means all the kernels
                 * completed, this is the special token we receive from the
                 * flush channel kernel that is issues at the end of the
                 * context */
                if (ma->cta_id_x == -1) {
                    done = true;
                    break;
                }

                std::stringstream ss;
                ss << "CTX " << HEX(ctx) << " - grid_launch_id "
                   << ma->grid_launch_id << " - CTA " << ma->cta_id_x << ","
                   << ma->cta_id_y << "," << ma->cta_id_z << " - warp "
                   << ma->warp_id << " - " << id_to_opcode_map[ma->opcode_id]
                   << " - " << "Size " << ma->size <<" - " << "MREF per threads(threadidx,data,address) : ";

                void* data_ptr;
                if(ma->size == 1)
                    data_ptr = (ma->config.data_8);
                else if(ma->size == 2)
                    data_ptr = (ma->config.data_16);
                else if(ma->size == 4)
                    data_ptr = (ma->config.data_32);
                else if(ma->size == 8)
                    data_ptr = (ma->config.data_64);
                else
                    data_ptr = (ma->config.data_128);

                for (int i = 0; i < 32; i++) {
                    ss << "Thread" << i << ",";

                    if(ma->size == 1) 
                        ss << HEX_DATA(*(static_cast<char*>(data_ptr) + i));
                    else if(ma->size == 2)
                        ss << HEX_DATA(*(static_cast<int16_t*>(data_ptr) + i));
                    else if(ma->size == 4)
                        ss << HEX_DATA(*(static_cast<int*>(data_ptr) + i));
                    else if(ma->size == 8) 
                        ss << "0x" << std::setfill('0') << std::setw(16) << std::hex << 
                        *(uint64_t*)(static_cast<double*>(data_ptr) + i);
                    else if(ma->size == 16) {
                        ss << "0x";
                        for(int j = 0; j < 16; j++) {
                            ss << std::setfill('0') << std::setw(2) << std::hex 
                                << (int)((static_cast<Data16*>(data_ptr) + i)->bytes[j]);
                        }
                    }
                    
                    /*
                    {
                        ss << "0x" << std::setfill('0') << std::setw(16)  << std::hex 
                            << *reinterpret_cast<unsigned long long*>((static_cast<double*>(data_ptr) + i)) << std::dec;
                    }
                    */

                    ss << "," << HEX(ma->config.addrs[i]) << " ";
                }

                printf("MEMTRACE: %s\n", ss.str().c_str());
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }
    free(recv_buffer);
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
    }
    CTXstate* ctx_state = new CTXstate;
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    ctx_state_map[ctx] = ctx_state;
    cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    if (verbose) {
        printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* flush channel */
    flush_channel<<<1, 1>>>(ctx_state->channel_dev);
    /* Make sure flush of channel is complete */
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("No CUDA error.\n");
    }
    assert(cudaGetLastError() == cudaSuccess);

    
    for(auto pair : free_address_map) {
            printf("free address : %llu\n", (pair.first));
    }
    
    /*
    printf("memcpy size : %d, alloc_size : %d, free_size : %d\n",memcpy_address_map.size(), alloc_address_map.size(), free_address_map.size());
    for(auto pair : alloc_address_map) {
        //printf("alloc address : %lld, size : %d\n", pair.first,pair.second);
        if((unsigned long long)pair.first != (*pair.first)) {
            printf("Also not Same!!! Raw : %lld, Ptr : %lld ",(unsigned long long)pair.first, *pair.first);
            printf("Same, alloc address : %lld, size : %d\n", *pair.first,pair.second);
        }
        else 
            printf("Same, alloc address : %lld, size : %d\n", pair.first,pair.second);
    }
    for(auto pair : free_address_map) {
        printf("free address : %lld\n", (pair.first));
    }
    for(auto pair : memcpy_address_map) {
        printf("memcpy address : %lld, size : %d\n", pair.first,pair.second);
    }
    */
    
    
    
    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    pthread_mutex_unlock(&mutex);
}
