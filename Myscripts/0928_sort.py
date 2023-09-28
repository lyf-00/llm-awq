with open('0928.sh','r') as f:
    cmds = f.read()

cmds = cmds.split('\n\n')
print(cmds[0])
cmds.remove('\n')
print(int(cmds[0].split('CUDA_VISIBLE_DEVICES=')[1][0]))
cmds.sort(key=lambda x: int(x.split('CUDA_VISIBLE_DEVICES=')[1][0]))
with open('0928_sort.sh','w') as f:
    f.write('\n'.join(cmds))