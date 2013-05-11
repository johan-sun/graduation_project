DEBUG ?= 
testClean = $(addprefix testcase/, $(shell ls testcase/ | egrep -v "*.yuv|encoder_baseline.cfg"))
export DEBUG
encode:
	$(MAKE) -C JMencoder
	rm -f testcase/lencod 2>/dev/null
	cp JMencoder/lencod testcase/.

.PHONY:clean
clean:
	$(MAKE) -C JMencoder clean
	rm -f $(testClean)

