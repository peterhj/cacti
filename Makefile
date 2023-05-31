include common.mk

.PHONY: all debug distclean

all:
	$(CARGO) build $(CARGO_TARGET) --release --lib --bins --examples

debug:
	$(CARGO) build $(CARGO_TARGET) --lib --bins --examples

distclean:
	rm -r target
