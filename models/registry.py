"""
Model registry and builders.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import os
import torch

from drugrec_benchmark.models.safedrug import SafeDrug
from drugrec_benchmark.models.premier import PREMIER
from drugrec_benchmark.models.refine import REFINE
from drugrec_benchmark.models.kamtl_medrec import (
	KAMTLMedRec,
	build_medical_cooccurrence,
	build_smiles_attribute_graph,
)
from drugrec_benchmark.models.leap import Leap
from drugrec_benchmark.models.gamenet import GAMENet
from drugrec_benchmark.models.retain import Retain
from drugrec_benchmark.models.micron import MICRON
from drugrec_benchmark.models.cognet import COGNet
from drugrec_benchmark.models.foursdrug import FourSDrug
from drugrec_benchmark.models.armr import ARMR
from drugrec_benchmark.models.raremed import RareMed
from drugrec_benchmark.models.sspnet import SSPNet
from drugrec_benchmark.models.vita import VITA
from drugrec_benchmark.models.ontopath import Ontopath, build_drug_paths, build_icd_paths
from drugrec_benchmark.models.compnet import CompNet
from drugrec_benchmark.models.carmen import Carmen, build_carmen_matrices
from drugrec_benchmark.models.molerec import (
	MoleRec,
	build_projection_and_smiles,
	graph_batch_from_smile,
)
from drugrec_benchmark.models.drechgr import (
	DRecHGR,
	build_hetero_adj,
	build_meta_path_edges,
	build_meta_path_graph,
	build_dgl_graph_from_adj,
)
from drugrec_benchmark.utils.dataset_utils import load_pickle
from drugrec_benchmark.utils.build_mpnn import build_mpnn
ModelBuilder = Callable[[Dict[str, Any], Dict[str, Any], torch.device], Tuple[Any, Dict[str, Any]]]

_MODEL_REGISTRY: Dict[str, ModelBuilder] = {}


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
	def decorator(func: ModelBuilder) -> ModelBuilder:
		_MODEL_REGISTRY[name.lower()] = func
		return func
	return decorator


def build_model(
	model_name: str,
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[Any, Dict[str, Any]]:
	builder = _MODEL_REGISTRY.get(model_name.lower())
	if not builder:
		raise NotImplementedError(f"Model '{model_name}' is not registered.")
	return builder(config, device)




@register_model("premier")
def build_premier(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[PREMIER, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ehr_adj_path = os.path.join(data_dir, config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"))

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)
	model = PREMIER(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		gat_heads=config["model"].get("gat_heads", 2),
		dropout=config["model"].get("dropout", 0.2),
		threshold=config["evaluation"].get("threshold", 0.5),
		multilabel_weight=config["model"].get("multilabel_weight", 0.05),
		ddi_weight=config["model"].get("ddi_weight", 0.0005),
		di_graph_weight=config["model"].get("di_graph_weight", 0.5),
		device=device,
	)
	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta

@register_model("safedrug")
def build_safedrug(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[SafeDrug, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ddi_mask_path = os.path.join(data_dir, config["dataset"]["ddi_mask_file"])
	molecule_path = os.path.join(data_dir, config["dataset"]["molecule_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

	ddi_adj = load_pickle(ddi_adj_path)
	ddi_mask_h = load_pickle(ddi_mask_path)
	molecule = load_pickle(molecule_path)

	mpnn_set, n_fingerprints, average_projection = build_mpnn(
		molecule,
		med_voc.idx2word,
		radius=config["model"].get("mpnn_radius", 2),
		device=device,
	)

	model = SafeDrug(
		vocab_size=vocab_size,
		ddi_adj=ddi_adj,
		ddi_mask_h=ddi_mask_h,
		mpnn_set=mpnn_set,
		n_fingerprints=n_fingerprints,
		average_projection=average_projection,
		emb_dim=config["model"].get("emb_dim", 64),
		target_ddi=config["model"].get("target_ddi", 0.06),
		kp=config["model"].get("kp", 0.05),
		threshold=config["evaluation"].get("threshold", 0.5),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("refine")
def build_refine(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[REFINE, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ehr_adj_path = os.path.join(data_dir, config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"))

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)

	model = REFINE(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 128),
		num_heads=config["model"].get("num_heads", 4),
		transformer_layers=config["model"].get("transformer_layers", 2),
		dropout=config["model"].get("dropout", 0.2),
		threshold=config["evaluation"].get("threshold", 0.5),
		gamma_bce=config["model"].get("gamma_bce", 0.90),
		gamma_hinge=config["model"].get("gamma_hinge", 0.05),
		beta=config["model"].get("beta", 0.50),
		bdi_scale=config["model"].get("bdi_scale", 1e-4),
		ddi_weight=config["model"].get("ddi_weight", 1.0),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("kamtl_medrec")
@register_model("kamtlmedrec")
@register_model("kamtl")
@register_model("medrec")
def build_kamtl_medrec(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[KAMTLMedRec, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	records_path = os.path.join(data_dir, config["dataset"]["records_file"])
	ehr_adj_path = os.path.join(data_dir, config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"))
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	molecule_path = os.path.join(data_dir, config["dataset"].get("molecule_file", "SMILES.pkl"))

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

	records = load_pickle(records_path)
	diag_med_adj, proc_med_adj = build_medical_cooccurrence(records, vocab_size)
	ehr_adj = load_pickle(ehr_adj_path)
	ddi_adj = load_pickle(ddi_adj_path)
	molecule = load_pickle(molecule_path)
	attr_adj = build_smiles_attribute_graph(
		molecule,
		med_voc.idx2word,
		top_k=config["model"].get("attr_top_k", 20),
	)

	model = KAMTLMedRec(
		vocab_size=vocab_size,
		diag_med_adj=diag_med_adj,
		proc_med_adj=proc_med_adj,
		ehr_adj=ehr_adj,
		attr_adj=attr_adj,
		emb_dim=config["model"].get("emb_dim", 128),
		dropout=config["model"].get("dropout", 0.3),
		threshold=config["evaluation"].get("threshold", 0.5),
		aux_weight=config["model"].get("aux_weight", 0.05),
		ddi_weight=config["model"].get("ddi_weight", 0.0),
		ddi_adj=ddi_adj,
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
		"diag_med_adj": diag_med_adj,
		"proc_med_adj": proc_med_adj,
		"attr_adj": attr_adj,
	}
	return model, meta

@register_model("leap")
def build_leap(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[Leap, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)

	model = Leap(
		vocab_size=vocab_size,
		emb_dim=config["model"].get("emb_dim", 64),
		max_len=config["model"].get("max_len", 20),
		threshold=config["evaluation"].get("threshold", 0.5),
		ddi_adj=ddi_adj,
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("raremed")
def build_raremed(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[RareMed, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)

	model = RareMed(
		vocab_size=vocab_size,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 512),
		encoder_layers=config["model"].get("encoder_layers", 3),
		nhead=config["model"].get("nhead", 4),
		dropout=config["model"].get("dropout", 0.3),
		threshold=config["evaluation"].get("threshold", 0.5),
		weight_multi=config["model"].get("weight_multi", 0.005),
		weight_ddi=config["model"].get("weight_ddi", 0.1),
		ddi_scale=config["model"].get("ddi_scale", 5e-4),
		patient_separate=config["model"].get("patient_separate", False),
		train_all_visits=config["model"].get("train_all_visits", False),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta

@register_model("gamenet")
def build_gamenet(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[GAMENet, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ehr_adj_path = os.path.join(data_dir, config["dataset"]["ehr_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)

	model = GAMENet(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		ddi_in_memory=config["model"].get("ddi_in_memory", True),
		target_ddi=config["model"].get("target_ddi", 0.06),
		temperature=config["model"].get("temperature", 2.0),
		decay_weight=config["model"].get("decay_weight", 0.85),
		threshold=config["evaluation"].get("threshold", 0.5),
		use_ddi=config["model"].get("use_ddi", True),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("retain")
def build_retain(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[Retain, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)

	model = Retain(
		vocab_size=vocab_size,
		emb_dim=config["model"].get("emb_dim", 64),
		ddi_adj=load_pickle(ddi_adj_path),
		threshold=0.4,
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("micron")
def build_micron(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[MICRON, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)

	model = MICRON(
		vocab_size=vocab_size,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		threshold=config["evaluation"].get("threshold", 0.5),
		alpha_current=config["model"].get("alpha_current", 0.75),
		multilabel_weight=config["model"].get("multilabel_weight", 5e-2),
		lambda_bce=config["model"].get("lambda_bce", 0.25),
		lambda_multi=config["model"].get("lambda_multi", 0.25),
		lambda_ddi=config["model"].get("lambda_ddi", 0.25),
		lambda_rec=config["model"].get("lambda_rec", 0.25),
		target_ddi=config["model"].get("target_ddi", 0.08),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("molerec")
def build_molerec(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[MoleRec, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ddi_mask_path = os.path.join(data_dir, config["dataset"]["ddi_mask_file"])
	molecule_path = os.path.join(data_dir, config["dataset"]["molecule_file"])
	substructure_path = os.path.join(
		data_dir,
		config["dataset"].get("substructure_file", "substructure_smiles.pkl"),
	)

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)

	ddi_adj = load_pickle(ddi_adj_path)
	ddi_mask_h = load_pickle(ddi_mask_path)
	molecule = load_pickle(molecule_path)
	average_projection, smiles_list = build_projection_and_smiles(molecule, med_voc.idx2word)
	average_projection = average_projection.to(device)
	mol_graphs = graph_batch_from_smile(smiles_list).to(device)

	use_embedding = bool(config["model"].get("use_embedding", False))
	substruct_graphs = None
	if not use_embedding:
		substruct_smiles = load_pickle(substructure_path)
		substruct_graphs = graph_batch_from_smile(substruct_smiles).to(device)

	model = MoleRec(
		vocab_size=vocab_size,
		ddi_adj=ddi_adj,
		ddi_mask_h=ddi_mask_h,
		mol_graphs=mol_graphs,
		average_projection=average_projection,
		substruct_graphs=substruct_graphs,
		emb_dim=config["model"].get("emb_dim", 64),
		target_ddi=config["model"].get("target_ddi", 0.06),
		coef=config["model"].get("coef", 2.5),
		threshold=config["evaluation"].get("threshold", 0.5),
		use_embedding=use_embedding,
		dropout=config["model"].get("dp", 0.7),
		gnn_num_layer=config["model"].get("gnn_num_layer", 4),
		gnn_type=config["model"].get("gnn_type", "gin"),
		graph_pooling=config["model"].get("graph_pooling", "mean"),
		virtual_node=config["model"].get("virtual_node", False),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta

@register_model("cognet")
def build_cognet(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[COGNet, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ehr_adj_path = os.path.join(
		data_dir,
		config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"),
	)
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ddi_mask_path = os.path.join(data_dir, config["dataset"]["ddi_mask_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)

	model = COGNet(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		max_len=config["model"].get("max_len", 45),
		beam_size=config["model"].get("beam_size", 4),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("vita")
def build_vita(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[VITA, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ehr_adj_path = os.path.join(
		data_dir,
		config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"),
	)
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ddi_mask_path = os.path.join(data_dir, config["dataset"]["ddi_mask_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)
	ddi_mask_h = load_pickle(ddi_mask_path)

	model = VITA(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		di_mask_h=ddi_mask_h,
		emb_dim=config["model"].get("emb_dim", 64),
		max_len=config["model"].get("max_len", 45),
		beam_size=config["model"].get("beam_size", 4),
		max_diag_num=config["model"].get("max_diag_num", 39),
		max_proc_num=config["model"].get("max_proc_num", 32),
		max_med_num=config["model"].get("max_med_num", 56),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("sspnet")
def build_sspnet(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[SSPNet, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ehr_adj_path = os.path.join(
		data_dir,
		config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"),
	)
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)

	model = SSPNet(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		target_ddi=config["model"].get("target_ddi", 0.06),
		coef=config["model"].get("coef", 2.5),
		threshold=config["evaluation"].get("threshold", 0.5),
		dropout=config["model"].get("dropout", 0.7),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("4sdrug")
def build_4sdrug(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[FourSDrug, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	if "sym_voc" in voc:
		sym_voc = voc["sym_voc"]
	else:
		sym_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(sym_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)

	model = FourSDrug(
		n_sym=vocab_size[0],
		n_drug=vocab_size[2],
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		dropout=config["model"].get("dropout", 0.4),
		threshold=config["evaluation"].get("threshold", 0.5),
		entropy_weight=config["model"].get("entropy_weight", 0.5),
		ddi_weight=config["model"].get("ddi_weight", 1.0),
		ddi_scale=config["model"].get("ddi_scale", 1e-5),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("carmen")
def build_carmen(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[Carmen, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	records_path = os.path.join(data_dir, config["dataset"]["records_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
	ddi_adj = load_pickle(ddi_adj_path)
	records = load_pickle(records_path)
	med2diag, med2proc, ehr_adj = build_carmen_matrices(records, vocab_size)

	model = Carmen(
		vocab_size=vocab_size,
		med2diag=med2diag,
		med2proc=med2proc,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		dropout=config["model"].get("dropout", 0.5),
		max_visits=config["model"].get("max_visits", 2),
		threshold=config["evaluation"].get("threshold", 0.5),
		gamma_bce=config["model"].get("gamma_bce", 0.95),
		gamma_margin=config["model"].get("gamma_margin", 0.05),
		ddi_weight=config["model"].get("ddi_weight", 0.0),
		ddi_scale=config["model"].get("ddi_scale", 5e-4),
		use_ehr_aug=config["model"].get("use_ehr_aug", True),
		use_ddi_encoding=config["model"].get("use_ddi_encoding", True),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("compnet")
def build_compnet(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[CompNet, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ehr_adj_path = os.path.join(data_dir, config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"))

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
	ddi_adj = load_pickle(ddi_adj_path)
	if os.path.exists(ehr_adj_path):
		ehr_adj = load_pickle(ehr_adj_path)
	else:
		ehr_adj = torch.eye(vocab_size[2]).cpu().numpy()

	model = CompNet(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		gcn_layers=tuple(config["model"].get("gcn_layers", [64])),
		num_channels=config["model"].get("num_channels", 128),
		dropout=config["model"].get("dropout", 0.5),
		threshold=config["evaluation"].get("threshold", 0.5),
		ddi_weight=config["model"].get("ddi_weight", 0.0),
		ddi_scale=config["model"].get("ddi_scale", 1e-5),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("ontopath")
def build_ontopath(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[Ontopath, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))
	ddi_adj = load_pickle(ddi_adj_path)
	drug_paths, atc_num = build_drug_paths(med_voc)
	icd_paths, icd_num = build_icd_paths(diag_voc)

	model = Ontopath(
		vocab_size=vocab_size,
		drug_paths=drug_paths,
		icd_paths=icd_paths,
		atc_num=atc_num,
		icd_num=icd_num,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 64),
		dropout=config["model"].get("dropout", 0.1),
		bidirectional=config["model"].get("bidirectional", False),
		initializer=config["model"].get("initializer", "uniform"),
		predictor=config["model"].get("predictor", "dot"),
		transformer_order=config["model"].get("transformer_order", True),
		transformer_layers=config["model"].get("transformer_layers", 1),
		transformer_heads=config["model"].get("transformer_heads", 2),
		threshold=config["evaluation"].get("threshold", 0.5),
		ddi_weight=config["model"].get("ddi_weight", 0.0),
		ddi_scale=config["model"].get("ddi_scale", 1e-5),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
		"atc_num": atc_num,
		"icd_num": icd_num,
	}
	return model, meta


@register_model("armr")
def build_armr(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[ARMR, Dict[str, Any]]:
	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ehr_adj_path = os.path.join(
		data_dir,
		config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"),
	)

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (
		len(diag_voc.idx2word),
		len(pro_voc.idx2word),
		len(med_voc.idx2word),
	)
	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)

	model = ARMR(
		vocab_size=vocab_size,
		ehr_adj=ehr_adj,
		ddi_adj=ddi_adj,
		emb_dim=config["model"].get("emb_dim", 256),
		history_k=config["model"].get("history_k", 3),
		max_visits=config["model"].get("max_visits", 30),
		threshold=config["evaluation"].get("threshold", 0.5),
		blend_weight=config["model"].get("blend_weight", 0.7),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
	}
	return model, meta


@register_model("drechgr")
def build_drechgr(
	config: Dict[str, Any],
	device: torch.device,
) -> Tuple[DRecHGR, Dict[str, Any]]:
	"""Build DRecHGR model from config."""
	import scipy.sparse as sp

	data_dir = config["dataset"]["data_dir"]
	vocab_path = os.path.join(data_dir, config["dataset"]["vocab_file"])
	records_path = os.path.join(data_dir, config["dataset"]["records_file"])
	ddi_adj_path = os.path.join(data_dir, config["dataset"]["ddi_adj_file"])
	ehr_adj_path = os.path.join(data_dir, config["dataset"].get("ehr_adj_file", "ehr_adj_final.pkl"))

	voc = load_pickle(vocab_path)
	med_voc = voc["med_voc"]
	diag_voc = voc["diag_voc"]
	pro_voc = voc["pro_voc"]
	vocab_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

	records = load_pickle(records_path)
	ddi_adj = load_pickle(ddi_adj_path)
	ehr_adj = load_pickle(ehr_adj_path)

	n_patients = len(records)
	n_meds = vocab_size[2]
	n_diags = vocab_size[0]

	patient_id_to_idx = {id(record): i for i, record in enumerate(records)}

	adj1 = build_hetero_adj(records, n_patients, n_meds, item_idx=2, device=device)
	adj2 = build_hetero_adj(records, n_patients, n_diags, item_idx=0, device=device)

	pm_rows, pm_cols = [], []
	for p_idx, patient in enumerate(records):
		for visit in patient:
			if len(visit) > 2:
				for med in visit[2]:
					if 0 <= med < n_meds:
						pm_rows.append(p_idx)
						pm_cols.append(med)
	pm_csr = sp.csr_matrix(
		(np.ones(len(pm_rows), dtype=np.float32), (pm_rows, pm_cols)),
		shape=(n_patients, n_meds),
		dtype=np.float32,
	)

	pd_rows, pd_cols = [], []
	for p_idx, patient in enumerate(records):
		for visit in patient:
			if len(visit) > 0:
				for diag in visit[0]:
					if 0 <= diag < n_diags:
						pd_rows.append(p_idx)
						pd_cols.append(diag)
	pd_csr = sp.csr_matrix(
		(np.ones(len(pd_rows), dtype=np.float32), (pd_rows, pd_cols)),
		shape=(n_patients, n_diags),
		dtype=np.float32,
	)

	pmp_threshold = config["model"].get("pmp_threshold", 5)
	pdp_threshold = config["model"].get("pdp_threshold", 3)
	pmp_edges = build_meta_path_edges(pm_csr, threshold=pmp_threshold)
	pdp_edges = build_meta_path_edges(pd_csr, threshold=pdp_threshold)

	pmp_graph = build_meta_path_graph(pmp_edges, n_patients, device)
	pdp_graph = build_meta_path_graph(pdp_edges, n_patients, device)
	ddi_graph = build_dgl_graph_from_adj(ddi_adj, device)
	ehr_graph = build_dgl_graph_from_adj(ehr_adj, device)
	meta_graphs = [pmp_graph, pdp_graph, ddi_graph, ehr_graph]

	model = DRecHGR(
		vocab_size=vocab_size,
		n_patients=n_patients,
		adj1=adj1,
		adj2=adj2,
		meta_graphs=meta_graphs,
		patient_id_to_idx=patient_id_to_idx,
		featuredim=config["model"].get("featDim", 64),
		nhid=config["model"].get("nhid", 8),
		num_heads=config["model"].get("num_heads", [8]),
		dropout=config["model"].get("dropout", 0.6),
		gnn_layer=config["model"].get("gnn_layer", 2),
		keep_rate=config["model"].get("keepRate", 0.5),
		threshold=config["evaluation"].get("threshold", 0.5),
		device=device,
	)

	meta = {
		"vocab_size": vocab_size,
		"voc": voc,
		"n_patients": n_patients,
		"patient_id_to_idx": patient_id_to_idx,
	}
	return model, meta
