#!/bin/sh
set -eu -o pipefail
if [ -e "../.gitmodules" ] ; then
  exec ../bootstrap.sh
fi
echo "- cacti: bootstrapping from git HEAD..."
cacti_remote_prefix="https://git.sr.ht/~ptrj"
echo "- cacti: cloning recursive dependencies via remote url ${cacti_remote_prefix}..."
git clone -b patch "${cacti_remote_prefix}/aho_corasick" ../aho_corasick
git clone "${cacti_remote_prefix}/byteorder" ../byteorder
git clone -b patch "${cacti_remote_prefix}/cc" ../cc
git clone -b patch "${cacti_remote_prefix}/cfg_if" ../cfg_if
git clone "${cacti_remote_prefix}/cmake" ../cmake
git clone -b patch "${cacti_remote_prefix}/crc32fast" ../crc32fast
git clone -b patch-0.26 "${cacti_remote_prefix}/futhark" ../futhark
git clone "${cacti_remote_prefix}/futhark_ffi" ../futhark_ffi
git clone "${cacti_remote_prefix}/futhark_syntax" ../futhark_syntax
git clone "${cacti_remote_prefix}/glob" ../glob
git clone -b patch "${cacti_remote_prefix}/half" ../half
git clone -b patch "${cacti_remote_prefix}/home" ../home
git clone "${cacti_remote_prefix}/libc" ../libc
git clone -b patch-0.7 "${cacti_remote_prefix}/libloading" ../libloading
git clone "${cacti_remote_prefix}/memchr" ../memchr
git clone -b patch "${cacti_remote_prefix}/minimal_lexical" ../minimal_lexical
git clone -b patch "${cacti_remote_prefix}/nom" ../nom
git clone -b patch "${cacti_remote_prefix}/once_cell" ../once_cell
git clone -b patch "${cacti_remote_prefix}/regex" ../regex
git clone -b patch "${cacti_remote_prefix}/repugnant_pickle" ../repugnant_pickle
git clone -b remaster "${cacti_remote_prefix}/rustc_serialize" ../rustc_serialize
git clone "${cacti_remote_prefix}/ryu" ../ryu
git clone -b patch "${cacti_remote_prefix}/safetensor_serialize" ../safetensor_serialize
git clone -b patch --recurse-submodules "${cacti_remote_prefix}/sentencepiece_ffi" ../sentencepiece_ffi
git clone -b patch "${cacti_remote_prefix}/smol_str" ../smol_str
git clone -b patch "${cacti_remote_prefix}/zip" ../zip
echo "- cacti: building and installing cacti-futhark to cabal bin path..."
make -C ../futhark install
echo "- cacti: bootstrap complete."
