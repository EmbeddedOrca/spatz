view -new wave
add wave -noupdate /tb_bin/i_dut/cluster_probe
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.dma_busy
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.r_stall
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.w_stall
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.r_valid
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.r_ready
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.w_valid
add wave -noupdate -group DMA /tb_bin/i_dut/i_cluster_wrapper/i_cluster/dma_events.w_ready
add wave -noupdate -group Snitch[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/stall
add wave -noupdate -group Snitch[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/lsu_stall
add wave -noupdate -group Snitch[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/acc_stall
add wave -noupdate -group Snitch[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/zero_lsb

add wave -noupdate -group Snitch[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/stall
add wave -noupdate -group Snitch[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/lsu_stall
add wave -noupdate -group Snitch[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/acc_stall
add wave -noupdate -group Snitch[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/zero_lsb

add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/busy_q
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/stall
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_valid_o
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_ready_i
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_snitch/is_branch
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/running_q
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/i_vfu/fpu_status_o
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_req_valid_o
add wave -noupdate -group Spatz[0] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[0]/i_spatz_cc/i_spatz/spatz_mem_req_ready_i

add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/busy_q
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/stall
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_valid_o
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/vfu_rsp_ready_i
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_snitch/is_branch
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/running_q
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/i_vfu/fpu_status_o
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_req_valid_o
add wave -noupdate -group Spatz[1] /tb_bin/i_dut/i_cluster_wrapper/i_cluster/gen_core[1]/i_spatz_cc/i_spatz/spatz_mem_req_ready_i
