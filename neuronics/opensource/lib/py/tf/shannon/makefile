ENTROPY_OP_DIR = entropy
JOINTENTROPY_OP_DIR = joint_entropy
CONDITIONALENTROPY_OP_DIR = conditional_entropy
MUTUALINFORMATION_OP_DIR = mutual_information

.PHONY:	subcompile

subcompile:
	$(MAKE) -C $(ENTROPY_OP_DIR) cpu
	$(MAKE) -C $(JOINTENTROPY_OP_DIR) cpu
	$(MAKE) -C $(CONDITIONALENTROPY_OP_DIR) cpu
	$(MAKE) -C $(MUTUALINFORMATION_OP_DIR) cpu
