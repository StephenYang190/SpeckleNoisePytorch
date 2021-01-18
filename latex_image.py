def generate_ablation_code(n, m):
    f = open('out.txt', 'w')

    linew = {
        '3': '0.3',
        '4': '0.24',
        '5': '0.19',
        '6': '0.16',
        '7': '0.14'
    }
    w = linew[str(n)]

    #name 为每一列图片下面显示的标注
    #range(1,7) 中 7表示我有6列 
    f.write("\\begin{figure}[t]\n")
    f.write('\\centering\n')
    for i in range(n):
        f.write('\\subfigure[]{')
        f.write("\\begin{minipage}{%s\\linewidth}\n" % w)
        #range(1,7) 中 7表示每列我有6张图 
        for j in range(m):
           #之前图片的命名在这边有大作用
            f.write('\\includegraphics[width=1\\textwidth]{}\n')
            f.write('\\vspace{3pt}\n')

        #打印图片下面显示的标注
        f.write("\\end{minipage}\n")
        f.write('}\n')
    f.write("\\end{figure}\n")

generate_ablation_code(6, 4)
