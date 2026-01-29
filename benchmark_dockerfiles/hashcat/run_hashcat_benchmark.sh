#!/bin/bash
set -e

# Default is Ethash
HASH_TYPE=${HASH_TYPE:-1400}

hashcat --benchmark --hash-type $HASH_TYPE --machine-readable --quiet
