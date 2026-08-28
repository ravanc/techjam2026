# Builds the os_signpost shim used by signposts.py.
DYLIB := libtechjam_signpost.dylib

all: $(DYLIB)

$(DYLIB): src/signpost_shim.c
	xcrun clang -O2 -dynamiclib -o $@ $<

clean:
	rm -rf $(DYLIB) traces

.PHONY: all clean
