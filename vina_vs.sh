mkdir LOGS
mkdir OUTS

for x in $(ls *pdbqt); do
	x='basename $x' echo $x
	vina --config config*txt --ligand $x --log LOGS/$x.log --out OUTS/$x.pdbqt
done
