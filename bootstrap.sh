#!/bin/bash
set -eu -o pipefail
if [ -e "../.gitmodules" ] ; then
  exec echo "- cacti: warning: submodule checkout, bailing"
fi
cacti_remote_prefix="https://github.com/peterhj"

echo "- cacti: Bootstrapping from git HEAD..."
echo "- cacti: Cloning recursive dependencies via remote url ${cacti_remote_prefix}..."
set -x
git clone -b patch "${cacti_remote_prefix}/aho_corasick" ../aho_corasick
git clone "${cacti_remote_prefix}/arrayvec" ../arrayvec
git clone -b patch "${cacti_remote_prefix}/blake2" ../blake2
git clone "${cacti_remote_prefix}/byteorder" ../byteorder
git clone -b patch "${cacti_remote_prefix}/cc" ../cc
git clone "${cacti_remote_prefix}/cell_split" ../cell_split
git clone -b patch "${cacti_remote_prefix}/cfg_if" ../cfg_if
git clone -b patch "${cacti_remote_prefix}/cmake" ../cmake
git clone "${cacti_remote_prefix}/constant_time_eq" ../constant_time_eq
git clone -b patch "${cacti_remote_prefix}/crc32fast" ../crc32fast
git clone -b patch-0.26 "${cacti_remote_prefix}/futhark" ../futhark
git clone "${cacti_remote_prefix}/futhark_ffi" ../futhark_ffi
git clone -b patch "${cacti_remote_prefix}/futhark_syntax" ../futhark_syntax
git clone -b patch "${cacti_remote_prefix}/fxhash" ../fxhash
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
git clone -b patch "${cacti_remote_prefix}/rustc_serialize" ../rustc_serialize
git clone -b patch --recurse-submodules "${cacti_remote_prefix}/sentencepiece_ffi" ../sentencepiece_ffi
git clone -b patch "${cacti_remote_prefix}/smol_str" ../smol_str
git clone -b patch "${cacti_remote_prefix}/zip" ../zip
{ set +x;} 2>&-

echo "- cacti: Configuring cacti-futhark (note: this calls 'cabal update')..."
set -x
make -C ../futhark configure
{ set +x;} 2>&-

echo "- cacti: Building and installing cacti-futhark to cabal bin path (this will take a while)..."
set -x
make -C ../futhark install
{ set +x;} 2>&-

echo "- cacti: Bootstrap complete."
echo "- cacti: Now, you may build cacti for development by running 'make'."
echo "- cacti: Or, if you would like a release build, run 'make release'."
