include common.mk

.PHONY: all build debug distclean

all: debug

build:
	$(CARGO) build $(CARGO_TARGET) --release --lib --bins

debug:
	$(CARGO) build $(CARGO_TARGET) --lib --bins

distclean:
	rm -r target
