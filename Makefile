DEBUG ?= 
testClean = $(addprefix testcase/, $(shell ls testcase/ | egrep -v "foreman_part_qcif.yuv|encoder_baseline.cfg"))
export DEBUG
encode:
	$(MAKE) -C JMencoder
	cp JMencoder/lencod testcase/.

.PHONY:clean
clean:
	$(MAKE) -C JMencoder clean
	rm -f $(testClean)

