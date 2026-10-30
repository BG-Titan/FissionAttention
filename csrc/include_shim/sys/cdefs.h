/* SPDX-License-Identifier: Apache-2.0
 * Include shim: CUDA <= 12.9 declares sinpi/cospi/tanpi/... without a
 * noexcept specification; glibc >= 2.41 (Ubuntu 25+) declares them via
 * __THROW == noexcept(true) under __GLIBC_USE(IEC_60559_FUNCS_EXT_C23), so
 * a .cu TU that reaches both headers fails with
 *   "error: exception specification is incompatible with that of previous
 *    function cospi".
 *
 * Neutralizing __THROW (empty) after glibc's own <sys/cdefs.h> runs makes
 * every glibc math declaration plain, matching CUDA's. We cannot -D__THROW=
 * because sys/cdefs.h re-defines it unconditionally; this shim is layered
 * in with -isystem so <sys/cdefs.h> resolves here first, defers to the real
 * header, then drops the noexcept macros.
 *
 * Used only for building fusew4's kernels (setup.py adds
 * -isystem csrc/include_shim). Not a vLLM/torch-wide patch.
 */
#include_next <sys/cdefs.h>

#ifdef __THROW
#undef __THROW
#endif
#define __THROW

#ifdef __THROWNL
#undef __THROWNL
#endif
#define __THROWNL

#ifdef __NTH
#undef __NTH
#endif
#define __NTH(fct) fct

#ifdef __NTHNL
#undef __NTHNL
#endif
#define __NTHNL(fct) fct
