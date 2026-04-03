import argparse
import numpy as np
import torch
import torch.nn.functional as F

from Dataloader import dataloader
from VID_Trans_model import VID_Trans


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f'Note: number of gallery samples is quite small, got {num_g}')

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


def extract_clip_feature(model, imgs, use_gpu=True, flip_test=False, pool='avg'):
    """
    imgs shape after squeeze(0): [num_clips, seq_len, 3, H, W] or [num_clips, ...]
    model returns feature for each clip
    """
    if use_gpu:
        imgs = imgs.cuda(non_blocking=True)

    num_clips = imgs.size(0)

    feats = model(imgs, None)
    feats = feats.view(num_clips, -1)

    if flip_test:
        flipped_imgs = torch.flip(imgs, dims=[-1])  # horizontal flip on width
        feats_flip = model(flipped_imgs, None)
        feats_flip = feats_flip.view(num_clips, -1)
        feats = 0.5 * (feats + feats_flip)

    if pool == 'avg':
        feats = torch.mean(feats, dim=0, keepdim=True)
    else:
        feats, _ = torch.max(feats, dim=0, keepdim=True)

    feats = l2_normalize(feats)
    return feats.squeeze(0).cpu()


def extract_features(model, loader, use_gpu=True, flip_test=False, pool='avg', desc='query'):
    feats, pids, camids = [], [], []

    model.eval()
    with torch.no_grad():
        for _, (imgs, batch_pids, batch_camids, _) in enumerate(loader):
            imgs = imgs.squeeze(0)
            feat = extract_clip_feature(
                model=model,
                imgs=imgs,
                use_gpu=use_gpu,
                flip_test=flip_test,
                pool=pool
            )
            feats.append(feat)
            pids.append(int(batch_pids[0]))
            camids.append(int(batch_camids.view(-1)[0].item()))

    feats = torch.stack(feats, dim=0)
    feats = l2_normalize(feats)
    pids = np.asarray(pids)
    camids = np.asarray(camids)

    print(f'Extracted features for {desc} set, obtained {feats.size(0)}-by-{feats.size(1)} matrix')
    return feats, pids, camids


def compute_distmat(qf, gf):
    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    return distmat.cpu().numpy()


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    Standard k-reciprocal re-ranking.
    Adapted for person re-id style evaluation.
    """
    original_dist = np.concatenate(
        [
            np.concatenate([q_q_dist, q_g_dist], axis=1),
            np.concatenate([q_g_dist.T, g_g_dist], axis=1),
        ],
        axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1.0 * original_dist / np.max(original_dist, axis=0))

    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2.0 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    original_dist = original_dist[:query_num, :]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j, ind in enumerate(indNonZero):
            temp_min[0, indImages[j]] += np.minimum(V[i, ind], V[indImages[j], ind])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    final_dist = final_dist[:, query_num:]
    return final_dist


def print_results(tag, distmat, q_pids, g_pids, q_camids, g_camids):
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print(f'{tag} Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    print('Rank-1  : {:.1%}'.format(cmc[0]))
    if len(cmc) > 4:
        print('Rank-5  : {:.1%}'.format(cmc[4]))
    if len(cmc) > 9:
        print('Rank-10 : {:.1%}'.format(cmc[9]))
    if len(cmc) > 19:
        print('Rank-20 : {:.1%}'.format(cmc[19]))
    return cmc[0], mAP


def test(model, queryloader, galleryloader, pool='avg', use_gpu=True,
         flip_test=False, rerank=False, k1=20, k2=6, lambda_value=0.3):
    qf, q_pids, q_camids = extract_features(
        model, queryloader, use_gpu=use_gpu, flip_test=flip_test, pool=pool, desc='query'
    )
    gf, g_pids, g_camids = extract_features(
        model, galleryloader, use_gpu=use_gpu, flip_test=flip_test, pool=pool, desc='gallery'
    )

    print('Computing original distance matrix')
    distmat = compute_distmat(qf, gf)
    rank1, mAP = print_results('Original', distmat, q_pids, g_pids, q_camids, g_camids)

    if rerank:
        print('Computing re-ranking distance matrix')
        q_q_dist = compute_distmat(qf, qf)
        g_g_dist = compute_distmat(gf, gf)
        rerank_dist = re_ranking(
            q_g_dist=distmat,
            q_q_dist=q_q_dist,
            g_g_dist=g_g_dist,
            k1=k1,
            k2=k2,
            lambda_value=lambda_value
        )
        rr_rank1, rr_mAP = print_results('Re-ranked', rerank_dist, q_pids, g_pids, q_camids, g_camids)
        return {
            'original_rank1': rank1,
            'original_mAP': mAP,
            'rerank_rank1': rr_rank1,
            'rerank_mAP': rr_mAP,
        }

    return {
        'original_rank1': rank1,
        'original_mAP': mAP,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VID-Trans-ReID test with flip TTA and re-ranking')
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help='Path to saved best checkpoint')
    parser.add_argument('--pool', default='avg', choices=['avg', 'max'])
    parser.add_argument('--flip_test', action='store_true', help='Use horizontal flip test-time augmentation')
    parser.add_argument('--rerank', action='store_true', help='Use k-reciprocal re-ranking')
    parser.add_argument('--k1', default=20, type=int)
    parser.add_argument('--k2', default=6, type=int)
    parser.add_argument('--lambda_value', default=0.3, type=float)
    args = parser.parse_args()

    _, _, num_classes, camera_num, _, q_val_loader, g_val_loader = dataloader(args.Dataset_name)

    model = VID_Trans(
        num_classes=num_classes,
        camera_num=camera_num,
        pretrainpath=None
    ).cuda()

    checkpoint = torch.load(args.model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    model.load_state_dict(checkpoint, strict=True)

    results = test(
        model=model,
        queryloader=q_val_loader,
        galleryloader=g_val_loader,
        pool=args.pool,
        use_gpu=True,
        flip_test=args.flip_test,
        rerank=args.rerank,
        k1=args.k1,
        k2=args.k2,
        lambda_value=args.lambda_value,
    )

    print('\nFinal Summary ----------')
    print('Original  -> Rank-1: {:.4f}, mAP: {:.4f}'.format(
        results['original_rank1'], results['original_mAP']
    ))
    if 'rerank_rank1' in results:
        print('Re-ranked -> Rank-1: {:.4f}, mAP: {:.4f}'.format(
            results['rerank_rank1'], results['rerank_mAP']
        ))
