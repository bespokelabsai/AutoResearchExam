#!/bin/sh
set -eu

mkdir -p /app/output
exec "$@"
