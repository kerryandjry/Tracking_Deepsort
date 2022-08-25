#!/bin/bash

BUILDDir=build

if [ -d "$BUILDDir" ]; then
    rm -rf $BUILDDir
fi

cmake -B $BUILDDir

cd $BUILDDir

make -j8
