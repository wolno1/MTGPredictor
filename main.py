from logic import (
    CardDatabase,
    META_DECKS_DIR,
    REGISTERED_DECKS_DIR,
    analyze_battlefield,
    recommend_blocks,
    recommend_combat_line,
    ensure_learning_file,
    load_decks_from_folder,
    parse_decklist,
    rank_matching_decks,
    record_learning_event,
    save_imported_deck,
    simulate_many_matchups,
    simulate_many_opening_hands,
    simulate_matchup,
    simulate_opening_hand,
    simulate_opening_turns,
    suggest_plays_from_hand,
    summarize_deck,
    train_from_matchup_results,
)


def _read_multiline(prompt: str, end_marker: str = "END") -> str:
    print(prompt)
    print(f"Escribe {end_marker} en una línea nueva para terminar.")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip().upper() == end_marker:
            break
        lines.append(line)
    return "\n".join(lines)


def _print_deck_list(decks: list) -> None:
    if not decks:
        print("No hay decks cargados.")
        return

    for index, deck in enumerate(decks, start=1):
        summary = summarize_deck(deck)
        colors = "".join(summary["colors"]) or "?"
        print(f"[{index}] {deck.name} | {summary['total_cards']} cartas | colores: {colors} | desconocidas: {len(summary['unknown_cards'])}")


def _choose_deck(decks: list):
    _print_deck_list(decks)
    if not decks:
        return None

    choice = input("Selecciona el número del deck: ").strip()
    if not choice.isdigit():
        print("Entrada inválida.")
        return None

    position = int(choice) - 1
    if position < 0 or position >= len(decks):
        print("Ese deck no existe.")
        return None

    return decks[position]


def _inspect_deck(decks: list) -> None:
    deck = _choose_deck(decks)
    if deck is None:
        return

    summary = summarize_deck(deck)
    print(f"\nDeck: {deck.name}")
    print(f"Total de cartas: {summary['total_cards']}")
    print(f"Colores detectados: {', '.join(summary['colors']) or 'desconocidos'}")
    print(f"CMC promedio estimado: {summary['average_mana_value']}")
    if summary["unknown_cards"]:
        print("Cartas no encontradas en la base de datos:")
        for card_name in summary["unknown_cards"]:
            print(f"- {card_name}")

    print("\nLista principal:")
    for entry in deck.entries:
        print(f"- {entry.quantity} {entry.name}")


def _infer_opponent(meta_decks: list, card_db: CardDatabase) -> None:
    seen_text = _read_multiline("Pega las cartas que has visto del rival, una por línea.")
    observed = parse_decklist(seen_text, deck_name="Observed", card_db=card_db)
    observed_cards = [entry.name for entry in observed.entries]

    if not observed_cards:
        print("No se detectaron cartas válidas.")
        return

    matches = rank_matching_decks(observed_cards, meta_decks, top_n=3)
    print("\nDecks más probables:")
    for match in matches:
        hits = ", ".join(match["hits"]) if match["hits"] else "sin coincidencias"
        print(f"- {match['deck_name']} | score: {match['score']} | cartas vistas: {hits}")

    record_learning_event("opponent_inference", {"observed_cards": observed_cards, "matches": matches})


def _ask_int(prompt: str, default: int = 0) -> int:
    raw_value = input(prompt).strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        print("Número inválido, usaré el valor por defecto.")
        return default


def _advise_play(meta_decks: list, card_db: CardDatabase) -> None:
    available_mana = _ask_int("¿Cuánto maná tienes disponible este turno? ", default=0)
    my_life = _ask_int("¿Cuántas vidas tienes? ", default=20)
    opponent_life = _ask_int("¿Cuántas vidas tiene el rival? ", default=20)
    land_played = input("¿Ya jugaste tierra este turno? [s/n]: ").strip().casefold() == "s"
    pressure = input("Presión del rival [low/medium/high, Enter para auto]: ").strip().casefold() or "medium"

    hand_text = _read_multiline("Pega tu mano actual, una carta por línea o con cantidades.")
    my_battlefield = _read_multiline("Pega tu mesa actual si quieres análisis de combate. Si está vacía, solo escribe END.")
    opponent_battlefield = _read_multiline("Pega la mesa del rival si la conoces. Si está vacía, solo escribe END.")
    observed_text = _read_multiline("Pega las cartas vistas del rival si quieres inferir su deck. Si no, deja vacío.")

    battlefield_summary = analyze_battlefield(
        my_battlefield=my_battlefield,
        opponent_battlefield=opponent_battlefield,
        my_life=my_life,
        opponent_life=opponent_life,
        card_db=card_db,
    )

    learning_state = ensure_learning_file()
    active_weights = learning_state.get("weights", {})

    suggestions = suggest_plays_from_hand(
        hand_text,
        available_mana=available_mana,
        card_db=card_db,
        lands_played_this_turn=land_played,
        opponent_pressure=pressure,
        top_n=3,
        my_battlefield=my_battlefield,
        opponent_battlefield=opponent_battlefield,
        my_life=my_life,
        opponent_life=opponent_life,
        learned_weights=active_weights,
    )

    combat_line = recommend_combat_line(
        my_battlefield=my_battlefield,
        opponent_battlefield=opponent_battlefield,
        my_life=my_life,
        opponent_life=opponent_life,
        card_db=card_db,
    )
    block_plan = recommend_blocks(
        my_battlefield=my_battlefield,
        opponent_attackers=opponent_battlefield,
        my_life=my_life,
        card_db=card_db,
    )

    print("\nResumen de combate:")
    print(f"- Ataque estimado tuyo: {battlefield_summary['my_attack']}")
    print(f"- Ataque estimado rival: {battlefield_summary['opponent_attack']}")
    print(f"- Riesgo actual: {battlefield_summary['danger_level']}")
    print(f"- Recomendación: {battlefield_summary['combat_recommendation']}")
    print(f"- Plan sugerido: {combat_line['summary']}")
    print(f"- Bloqueos sugeridos: {block_plan['summary']}")
    for assignment in block_plan.get("assignments", []):
        print(f"  • {assignment}")

    print("\nPesos activos de aprendizaje:")
    for key, value in active_weights.items():
        print(f"- {key}: {value}")

    print("\nMejores jugadas sugeridas:")
    for index, suggestion in enumerate(suggestions, start=1):
        print(f"{index}. {suggestion['name']} | score {suggestion['score']} | {suggestion['reason']}")
        if suggestion.get("note"):
            print(f"   Nota: {suggestion['note']}")
        if suggestion.get("combat_summary") and index == 1:
            print(f"   Combate: {suggestion['combat_summary']}")

    observed = parse_decklist(observed_text, deck_name="Observed", card_db=card_db)
    observed_cards = [entry.name for entry in observed.entries]
    if observed_cards:
        matches = rank_matching_decks(observed_cards, meta_decks, top_n=3)
        print("\nDeck rival más probable:")
        for match in matches:
            print(f"- {match['deck_name']} | score {match['score']} | coincidencias: {', '.join(match['hits']) or 'ninguna'}")
    else:
        matches = []

    record_learning_event(
        "play_advice",
        {
            "available_mana": available_mana,
            "my_life": my_life,
            "opponent_life": opponent_life,
            "pressure": pressure,
            "battlefield_summary": battlefield_summary,
            "suggestions": suggestions,
            "observed_cards": observed_cards,
            "matches": matches,
        },
    )


def _simulate_opening(meta_decks: list, card_db: CardDatabase) -> None:
    print("\nSimulación de mano inicial")
    deck = _choose_deck(meta_decks)
    if deck is None:
        return

    seed = _ask_int("Semilla opcional para repetir la simulación [Enter = aleatoria]: ", default=0)
    result = simulate_opening_hand(deck, card_db=card_db, seed=None if seed == 0 else seed)

    print(f"\nDeck: {deck.name}")
    print("Mano inicial simulada:")
    for card_name in result["hand"]:
        print(f"- {card_name}")
    print(f"Tierras: {result['land_count']}")
    print(f"Curva media: {result['average_mana_value']}")
    print(f"Jugadas tempranas casteables: {result['castable_early_plays']}")
    print(f"Decisión sugerida: {result['decision'].upper()}")
    print(f"Motivo: {result['reason']}")

    record_learning_event(
        "opening_hand_simulation",
        {
            "deck": deck.name,
            "result": result,
        },
    )


def _simulate_many_openings(meta_decks: list, card_db: CardDatabase) -> None:
    print("\nSimulación de muchas manos")
    deck = _choose_deck(meta_decks)
    if deck is None:
        return

    simulations = _ask_int("¿Cuántas manos quieres simular? ", default=100)
    seed = _ask_int("Semilla opcional para repetir resultados [Enter = aleatoria]: ", default=0)
    result = simulate_many_opening_hands(deck, card_db=card_db, simulations=simulations, seed=None if seed == 0 else seed)

    print(f"\nDeck: {deck.name}")
    print(f"Simulaciones: {result['simulations']}")
    print(f"Keep rate: {result['keep_rate']}%")
    print(f"Mulligan rate: {result['mulligan_rate']}%")
    print(f"Tierras promedio: {result['average_lands']}")
    print(f"Curva media promedio: {result['average_curve']}")
    print(f"Jugadas tempranas promedio: {result['average_early_plays']}")

    if result['example_keeps']:
        print("\nEjemplos de manos keep:")
        for hand in result['example_keeps']:
            print(f"- {', '.join(hand)}")

    if result['example_mulligans']:
        print("\nEjemplos de manos mulligan:")
        for hand in result['example_mulligans']:
            clean_hand = [card.replace('\n', ' ').strip() for card in hand]
            print(f"- {', '.join(clean_hand)}")

    record_learning_event(
        "many_opening_hand_simulations",
        {
            "deck": deck.name,
            "result": result,
        },
    )


def _simulate_opening_turns_cli(meta_decks: list, card_db: CardDatabase) -> None:
    print("\nSimulación automática de turnos 1-3")
    deck = _choose_deck(meta_decks)
    if deck is None:
        return

    turns = _ask_int("¿Cuántos turnos quieres simular? ", default=3)
    seed = _ask_int("Semilla opcional [Enter = aleatoria]: ", default=0)
    result = simulate_opening_turns(deck, card_db=card_db, turns=turns, seed=None if seed == 0 else seed)

    print(f"\nDeck: {deck.name}")
    for step in result["turn_log"]:
        print(f"\nTurno {step['turn']} | vidas {step['life']} - rival {step['opponent_life']}")
        for phase in step.get("phases", []):
            print(f"- {phase['step']}: {phase['detail']}")
            if 'dispara' in phase['detail'] or 'triggers:' in phase['detail']:
                print("  -> Hubo una habilidad disparada en esta ventana")
            if 'remueve' in phase['detail']:
                print("  -> El simulador eligió un objetivo prioritario")
        print(f"Resumen del turno: {step['play']} | mano restante {step['hand_size']}")
        if 'respuesta:' in step['play']:
            print("  -> Hubo interacción en la pila durante este turno")
            print(f"  -> Cadena: {step['play']}")

    print(f"\nTierras en juego: {result['lands_in_play']}")
    print(f"Tamaño final de mano: {result['final_hand_size']}")
    print(f"Mesa estimada: {', '.join(result['battlefield'])}")

    record_learning_event(
        "opening_turns_simulation",
        {
            "deck": deck.name,
            "result": result,
        },
    )


def _simulate_matchup_cli(meta_decks: list, card_db: CardDatabase) -> None:
    print("\nSimulación de enfrentamiento entre decks")
    print("Elige el Deck A")
    deck_a = _choose_deck(meta_decks)
    if deck_a is None:
        return

    print("\nElige el Deck B")
    deck_b = _choose_deck(meta_decks)
    if deck_b is None:
        return

    turns = _ask_int("¿Cuántos turnos quieres simular? ", default=3)
    seed = _ask_int("Semilla opcional [Enter = aleatoria]: ", default=0)
    result = simulate_matchup(deck_a, deck_b, card_db=card_db, turns=turns, seed=None if seed == 0 else seed)

    print(f"\nResultado: {result['summary']}")
    print(f"Ganador estimado: {result['winner']}")
    print(f"Vidas finales: {deck_a.name} {result['life_a']} | {deck_b.name} {result['life_b']}")

    print("\nHistorial del enfrentamiento:")
    for step in result['turn_history']:
        print(f"\nTurno {step['turn']} - {step['active_player']}")
        for phase in step.get('phases', []):
            print(f"- {phase['step']}: {phase['detail']}")
            if 'dispara' in phase['detail'] or 'triggers:' in phase['detail']:
                print("  -> Se registró una habilidad disparada")
            if 'remueve' in phase['detail']:
                print("  -> Se seleccionó un objetivo prioritario")
        print(f"Resumen: {step['play']} | daño {step['damage_dealt']} | vidas {step['life']} a {step['opponent_life']}")
        if 'respuesta:' in step['play']:
            print("  -> Se registró una respuesta sobre la pila")
            print(f"  -> Cadena: {step['play']}")

    record_learning_event(
        "matchup_simulation",
        {
            "deck_a": deck_a.name,
            "deck_b": deck_b.name,
            "result": result,
        },
    )


def _simulate_many_matchups_cli(meta_decks: list, card_db: CardDatabase) -> None:
    print("\nSimulación masiva de enfrentamientos")
    print("Elige el Deck A")
    deck_a = _choose_deck(meta_decks)
    if deck_a is None:
        return

    print("\nElige el Deck B")
    deck_b = _choose_deck(meta_decks)
    if deck_b is None:
        return

    simulations = _ask_int("¿Cuántos enfrentamientos quieres correr? ", default=100)
    turns = _ask_int("¿Cuántos turnos por partida? ", default=3)
    seed = _ask_int("Semilla opcional [Enter = aleatoria]: ", default=0)
    result = simulate_many_matchups(
        deck_a,
        deck_b,
        card_db=card_db,
        simulations=simulations,
        turns=turns,
        seed=None if seed == 0 else seed,
    )

    print(f"\n{deck_a.name} vs {deck_b.name}")
    print(f"Simulaciones: {result['simulations']}")
    print(f"Winrate {deck_a.name}: {result['winrate_a']}%")
    print(f"Winrate {deck_b.name}: {result['winrate_b']}%")
    print(f"Empates: {result['draw_rate']}%")
    print(f"Score medio {deck_a.name}: {result['average_score_a']}")
    print(f"Score medio {deck_b.name}: {result['average_score_b']}")

    learned_state = train_from_matchup_results(result)

    print("\nPesos actualizados:")
    for key, value in learned_state.get("weights", {}).items():
        print(f"- {key}: {value}")

    record_learning_event(
        "many_matchup_simulations",
        {
            "deck_a": deck_a.name,
            "deck_b": deck_b.name,
            "result": result,
        },
    )


def _show_learning_state() -> None:
    state = ensure_learning_file()
    print("\nPesos actuales de aprendizaje:")
    for key, value in state.get("weights", {}).items():
        print(f"- {key}: {value}")
    print(f"Eventos guardados: {len(state.get('history', []))}")


def _recommend_blocks_cli(card_db: CardDatabase) -> None:
    print("\nEvaluación de bloqueos defensivos")
    my_life = _ask_int("¿Cuántas vidas tienes? ", default=20)
    my_battlefield = _read_multiline("Pega tu mesa actual, una carta por línea.")
    opponent_attackers = _read_multiline("Pega los atacantes o la mesa rival que te presiona.")

    result = recommend_blocks(
        my_battlefield=my_battlefield,
        opponent_attackers=opponent_attackers,
        my_life=my_life,
        card_db=card_db,
    )

    print(f"\nPlan: {result['summary']}")
    for assignment in result.get("assignments", []):
        print(f"- {assignment}")

    record_learning_event(
        "block_recommendation",
        {
            "my_life": my_life,
            "result": result,
        },
    )


def _import_deck(card_db: CardDatabase) -> None:
    deck_name = input("Nombre del deck: ").strip() or "ImportedDeck"
    destination = input("Guardar en [1] meta_decks o [2] registered_decks: ").strip()
    target_folder = META_DECKS_DIR if destination == "1" else REGISTERED_DECKS_DIR
    deck_text = _read_multiline("Pega aquí la lista del deck.")

    path, deck = save_imported_deck(deck_text, deck_name, target_folder, card_db=card_db)
    summary = summarize_deck(deck)

    print(f"\nDeck guardado: {path.name}")
    print(f"Cartas principales: {summary['total_cards']}")
    if summary["unknown_cards"]:
        print("Atención: estas cartas no se encontraron en la base de datos:")
        for card_name in summary["unknown_cards"]:
            print(f"- {card_name}")

    record_learning_event(
        "deck_imported",
        {
            "name": deck_name,
            "path": str(path),
            "total_cards": summary["total_cards"],
            "unknown_cards": summary["unknown_cards"],
        },
    )


def main() -> None:
    print("MTG Predictor - Alchemy Tools")
    ensure_learning_file()

    card_db = CardDatabase()
    print("Cargando base de cartas...")
    card_db.load()
    meta_decks = load_decks_from_folder(META_DECKS_DIR, card_db=card_db)

    while True:
        print("\n¿Qué quieres hacer?")
        print("1. Ver meta decks cargados")
        print("2. Inspeccionar un deck")
        print("3. Inferir el deck rival")
        print("4. Importar un deck pegado desde un website")
        print("5. Sugerir la mejor jugada desde tu mano")
        print("6. Simular mano inicial y mulligan")
        print("7. Simular muchas manos seguidas")
        print("8. Simular turnos 1-3 automáticamente")
        print("9. Simular enfrentamiento entre dos decks")
        print("10. Simular muchos enfrentamientos")
        print("11. Ver pesos de aprendizaje")
        print("12. Evaluar bloqueos defensivos")
        print("13. Salir")

        option = input("Elige una opción: ").strip()
        if option == "1":
            meta_decks = load_decks_from_folder(META_DECKS_DIR, card_db=card_db)
            _print_deck_list(meta_decks)
        elif option == "2":
            _inspect_deck(meta_decks)
        elif option == "3":
            _infer_opponent(meta_decks, card_db)
        elif option == "4":
            _import_deck(card_db)
            meta_decks = load_decks_from_folder(META_DECKS_DIR, card_db=card_db)
        elif option == "5":
            _advise_play(meta_decks, card_db)
        elif option == "6":
            _simulate_opening(meta_decks, card_db)
        elif option == "7":
            _simulate_many_openings(meta_decks, card_db)
        elif option == "8":
            _simulate_opening_turns_cli(meta_decks, card_db)
        elif option == "9":
            _simulate_matchup_cli(meta_decks, card_db)
        elif option == "10":
            _simulate_many_matchups_cli(meta_decks, card_db)
        elif option == "11":
            _show_learning_state()
        elif option == "12":
            _recommend_blocks_cli(card_db)
        elif option == "13":
            print("Hasta luego.")
            break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
