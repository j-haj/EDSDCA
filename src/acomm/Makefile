
CC := g++-6
CLANG := clang++

CSTD := -std=c++14
CFLAGS := -Wall -pedantic


.PHONY: tests
tests:
	$(CC) $(CSTD) $(CFLAGS) queue_test.cc

.PHONY: debug
debug:
	$(CLANG) $(CSTD) $(CFLAGS) -g queue_test.cc

