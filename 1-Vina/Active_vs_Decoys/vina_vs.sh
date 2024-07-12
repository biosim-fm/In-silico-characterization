mkdir LOGS
mkdir OUTS

for x in $(ls *pdbqt); do
	x='basename $x' echo $x
	vina --config file.conf --ligand $x --log LOGS/$x.log --out OUTS/$x.pdbqt
done
