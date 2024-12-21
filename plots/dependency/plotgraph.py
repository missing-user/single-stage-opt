
objs = make_objs() 
prob = objectives.LeastSquaresProblem.from_tuples(objs)
latexplot.figure()
G, pos = prob.plot_graph(show=False)
from network2tikz import plot
plot(G, layout=pos, filename="dependency_graph.tex", vertex_label=G.vs['name'])

latexplot.savenshow("dependency_graph")