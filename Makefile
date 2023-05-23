include common.mk

.PHONY: all distclean

all:
	$(CARGO) build $(CARGO_TARGET) --release --lib --bins --examples

distclean:
	rm -r target
