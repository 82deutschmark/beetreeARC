def parse_finish_step(content):
    result = {
        "finish_data": content,
        "finish_status": None,
        "calls": [],
        "judge_stats": {}
    }
    
    if isinstance(content, dict):
        if "result" in content:
            result["finish_status"] = content["result"]
            
        candidates_obj = content.get("candidates_object", {})
        candidates_keys = list(candidates_obj.keys())

        # Extract judges info
        if "selection_details" in content:
            sel_details = content["selection_details"]
            if isinstance(sel_details, dict) and "judges" in sel_details:
                judges = sel_details["judges"]
                if isinstance(judges, dict):
                    for judge_name, judge_data in judges.items():
                        if not isinstance(judge_data, dict):
                            continue

                        duration = judge_data.get("duration_seconds", 0)
                        cost = judge_data.get("total_cost", 0)
                        input_tokens = judge_data.get("input_tokens", 0)
                        output_tokens = judge_data.get("output_tokens", 0)
                        cached_tokens = judge_data.get("cached_tokens", 0)
                        timing_breakdown = judge_data.get("timing_breakdown", [])
                        model = judge_data.get("model", "")
                        
                        display_name = f"Judge ({judge_name.capitalize()})"
                        if model:
                            display_name += f" - {model}"
                            
                        result["calls"].append({
                            "name": display_name,
                            "duration": duration,
                            "cost": cost,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "cached_tokens": cached_tokens,
                            "timing_breakdown": timing_breakdown,
                            "status": ""
                        })
                        
                        # Extract detailed stats
                        parsed = judge_data.get("parsed", {})
                        if parsed:
                            evaluations = parsed.get("candidates", [])
                            ranking = parsed.get("final_ranking_by_candidate", [])
                            
                            # Fallback: if ranking is missing, sort by score
                            if not ranking and evaluations:
                                sorted_evals = sorted(evaluations, key=lambda x: x.get("score", 0), reverse=True)
                                ranking = [e.get("candidate_id") for e in sorted_evals if "candidate_id" in e]
                            
                            stats_list = []
                            for eval_item in evaluations:
                                cid = eval_item.get("candidate_id")
                                score = eval_item.get("score", 0)
                                tier = eval_item.get("tier", "")
                                
                                if isinstance(cid, int) and 0 <= cid < len(candidates_keys):
                                    cand_key = candidates_keys[cid]
                                    actual_cand = candidates_obj[cand_key]
                                    is_correct = actual_cand.get("is_correct", False)
                                    
                                    # Determine rank
                                    rank = -1
                                    if cid in ranking:
                                        rank = ranking.index(cid) + 1
                                    
                                    stats_list.append({
                                        "is_correct": is_correct,
                                        "score": score,
                                        "rank": rank,
                                        "tier": tier
                                    })
                            
                            result["judge_stats"][judge_name] = {
                                "evaluations": stats_list,
                                "cost": cost,
                                "duration": duration
                            }
    return result
