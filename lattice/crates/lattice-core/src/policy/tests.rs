//! The §9.2 table, row by row, and every combination of its inputs against
//! the table written a second way (an ordered list of rows, first match wins).

use super::*;

const ALL_REFUSALS: [PathRefusal; 13] = [
    PathRefusal::Outside,
    PathRefusal::GitDir,
    PathRefusal::Ignored,
    PathRefusal::SecretDefault,
    PathRefusal::LatticeState,
    PathRefusal::Device,
    PathRefusal::Stream,
    PathRefusal::Unc,
    PathRefusal::ShortName,
    PathRefusal::RemoteLink,
    PathRefusal::TrailingDotOrSpace,
    PathRefusal::TooLong,
    PathRefusal::Escape,
];

const TOOLS: [ToolClass; 13] = [
    ToolClass::Read,
    ToolClass::AskQuestion,
    ToolClass::Stage,
    ToolClass::RunCommand,
    ToolClass::Mcp,
    ToolClass::Browser { sensitive: false },
    ToolClass::Browser { sensitive: true },
    ToolClass::Desktop { asks: false },
    ToolClass::Desktop { asks: true },
    ToolClass::Keep(KeepKind::Change),
    ToolClass::Keep(KeepKind::Restore),
    ToolClass::Keep(KeepKind::CommandUndo),
    ToolClass::KeepAll,
];

fn open() -> Gates {
    Gates {
        staged_waiting: 0,
        command_running: false,
        lease: Lease::Free,
    }
}

fn none() -> Standing {
    Standing::default()
}

fn matched() -> Standing {
    Standing {
        command_entry: Some("e_1".into()),
        mcp_allowlisted: false,
    }
}

fn path(class: PathClass) -> Target {
    Target::Path(class)
}

fn command(cwd: PathClass) -> Target {
    Target::Command { cwd }
}

fn agent(tool: ToolClass, target: &Target, gates: &Gates, standing: &Standing) -> Verdict {
    decide(Mode::Agent, true, tool, target, gates, standing)
}

#[test]
fn read_tools_allow_normal_and_authority_paths_in_every_mode_and_trust() {
    for mode in [Mode::Ask, Mode::Agent] {
        for trusted in [false, true] {
            for class in [PathClass::Normal, PathClass::Authority] {
                assert_eq!(
                    decide(
                        mode,
                        trusted,
                        ToolClass::Read,
                        &path(class),
                        &open(),
                        &none()
                    ),
                    Verdict::Allow(Because::Read)
                );
            }
        }
    }
}

#[test]
fn read_tools_refuse_every_refused_path_with_its_reason() {
    for refusal in ALL_REFUSALS {
        for mode in [Mode::Ask, Mode::Agent] {
            assert_eq!(
                decide(
                    mode,
                    true,
                    ToolClass::Read,
                    &path(PathClass::Refused(refusal)),
                    &open(),
                    &none()
                ),
                Verdict::Refuse(Reason::Path(refusal))
            );
        }
        let sentence = refusal.sentence();
        assert!(
            sentence.ends_with('.') && !sentence.is_empty(),
            "{sentence}"
        );
    }
}

#[test]
fn ask_question_is_always_allowed() {
    for mode in [Mode::Ask, Mode::Agent] {
        for trusted in [false, true] {
            assert_eq!(
                decide(
                    mode,
                    trusted,
                    ToolClass::AskQuestion,
                    &Target::None,
                    &open(),
                    &none()
                ),
                Verdict::Allow(Because::Question)
            );
        }
    }
}

#[test]
fn ask_mode_refuses_every_acting_tool_even_with_a_standing_match() {
    for tool in TOOLS
        .into_iter()
        .filter(|tool| !matches!(tool, ToolClass::Read | ToolClass::AskQuestion))
    {
        assert_eq!(
            decide(
                Mode::Ask,
                true,
                tool,
                &command(PathClass::Normal),
                &open(),
                &matched()
            ),
            Verdict::Refuse(Reason::AskMode),
            "{tool:?}"
        );
    }
    assert_eq!(
        Reason::AskMode.sentence(),
        "That is not available in Ask mode."
    );
}

#[test]
fn agent_mode_in_an_untrusted_folder_refuses_writes_and_commands() {
    for tool in [
        ToolClass::Stage,
        ToolClass::RunCommand,
        ToolClass::Keep(KeepKind::Change),
        ToolClass::KeepAll,
        ToolClass::Mcp,
    ] {
        assert_eq!(
            decide(
                Mode::Agent,
                false,
                tool,
                &path(PathClass::Normal),
                &open(),
                &matched()
            ),
            Verdict::Refuse(Reason::Untrusted)
        );
    }
    assert_eq!(
        Reason::Untrusted.sentence(),
        "Trust this folder to use Agent mode."
    );
}

#[test]
fn staging_follows_the_path_class_and_needs_no_lease() {
    for refusal in ALL_REFUSALS {
        assert_eq!(
            agent(
                ToolClass::Stage,
                &path(PathClass::Refused(refusal)),
                &open(),
                &none()
            ),
            Verdict::Refuse(Reason::Path(refusal))
        );
    }
    // Staging writes nothing, so neither the lease nor a running command stops it.
    let busy = Gates {
        staged_waiting: 4,
        command_running: true,
        lease: Lease::Elsewhere,
    };
    assert_eq!(
        agent(ToolClass::Stage, &path(PathClass::Normal), &busy, &none()),
        Verdict::Allow(Because::Stage)
    );
    assert_eq!(
        agent(
            ToolClass::Stage,
            &path(PathClass::Authority),
            &busy,
            &none()
        ),
        Verdict::Allow(Because::StageAuthority)
    );
}

#[test]
fn a_command_waits_for_staged_changes_even_when_a_standing_entry_matches() {
    let one = Gates {
        staged_waiting: 1,
        ..open()
    };
    for standing in [none(), matched()] {
        assert_eq!(
            agent(
                ToolClass::RunCommand,
                &command(PathClass::Normal),
                &one,
                &standing
            ),
            Verdict::Ask(AskKind::GateStaged { waiting: 1 })
        );
    }
    // The boundary: none waiting opens the gate.
    assert_eq!(
        agent(
            ToolClass::RunCommand,
            &command(PathClass::Normal),
            &open(),
            &matched()
        ),
        Verdict::Allow(Because::Standing {
            entry: "e_1".into()
        })
    );
    assert_eq!(
        agent(
            ToolClass::RunCommand,
            &command(PathClass::Normal),
            &open(),
            &none()
        ),
        Verdict::Ask(AskKind::Command)
    );
}

#[test]
fn one_command_per_folder_and_the_refusal_beats_the_gate() {
    let running = Gates {
        staged_waiting: 2,
        command_running: true,
        lease: Lease::Held,
    };
    for standing in [none(), matched()] {
        assert_eq!(
            agent(
                ToolClass::RunCommand,
                &command(PathClass::Normal),
                &running,
                &standing
            ),
            Verdict::Refuse(Reason::CommandRunning)
        );
    }
    assert!(Reason::CommandRunning.is_conflict());
}

#[test]
fn a_command_in_a_refused_folder_is_refused_with_the_folders_reason() {
    assert_eq!(
        agent(
            ToolClass::RunCommand,
            &command(PathClass::Refused(PathRefusal::GitDir)),
            &open(),
            &matched()
        ),
        Verdict::Refuse(Reason::Path(PathRefusal::GitDir))
    );
}

#[test]
fn mcp_tools_ask_unless_allowlisted() {
    let allowlisted = Standing {
        command_entry: None,
        mcp_allowlisted: true,
    };
    assert_eq!(
        agent(ToolClass::Mcp, &Target::None, &open(), &allowlisted),
        Verdict::Allow(Because::Mcp)
    );
    assert_eq!(
        agent(ToolClass::Mcp, &Target::None, &open(), &none()),
        Verdict::Ask(AskKind::Mcp)
    );
}

/// Auto mode on the desktop ("Money and accounts ask"): money and
/// accounts ask, everything else goes ahead; neither in Ask mode or an
/// untrusted folder.
#[test]
fn desktop_actions_in_auto_mode_ask_only_for_money_and_accounts() {
    assert_eq!(
        agent(
            ToolClass::Desktop { asks: false },
            &Target::None,
            &open(),
            &none()
        ),
        Verdict::Allow(Because::AutoMode)
    );
    assert_eq!(
        agent(
            ToolClass::Desktop { asks: true },
            &Target::None,
            &open(),
            &none()
        ),
        Verdict::Ask(AskKind::Desktop)
    );
    for asks in [false, true] {
        let tool = ToolClass::Desktop { asks };
        assert_eq!(
            decide(Mode::Ask, true, tool, &Target::None, &open(), &none()),
            Verdict::Refuse(Reason::AskMode)
        );
        assert_eq!(
            decide(Mode::Agent, false, tool, &Target::None, &open(), &none()),
            Verdict::Refuse(Reason::Untrusted)
        );
    }
}

/// The agent's browser: a view or an edit goes ahead, an effect another
/// person will see (or money, a deletion, an account) asks; neither in Ask
/// mode or an untrusted folder.
#[test]
fn browser_actions_ask_only_when_their_effect_does() {
    assert_eq!(
        agent(
            ToolClass::Browser { sensitive: false },
            &Target::None,
            &open(),
            &none()
        ),
        Verdict::Allow(Because::Browser)
    );
    assert_eq!(
        agent(
            ToolClass::Browser { sensitive: true },
            &Target::None,
            &open(),
            &none()
        ),
        Verdict::Ask(AskKind::Browser)
    );
    for sensitive in [false, true] {
        let tool = ToolClass::Browser { sensitive };
        assert_eq!(
            decide(Mode::Ask, true, tool, &Target::None, &open(), &none()),
            Verdict::Refuse(Reason::AskMode)
        );
        assert_eq!(
            decide(Mode::Agent, false, tool, &Target::None, &open(), &none()),
            Verdict::Refuse(Reason::Untrusted)
        );
    }
}

#[test]
fn keep_all_skips_authority_files_and_an_individual_keep_asks_natively() {
    let authority = path(PathClass::Authority);
    assert_eq!(
        agent(ToolClass::KeepAll, &authority, &open(), &none()),
        Verdict::Refuse(Reason::AuthorityKeptAlone)
    );
    assert!(!Reason::AuthorityKeptAlone.is_conflict());
    assert_eq!(
        agent(
            ToolClass::KeepAll,
            &path(PathClass::Normal),
            &open(),
            &none()
        ),
        Verdict::Allow(Because::Keep)
    );
    for kind in [KeepKind::Change, KeepKind::Restore, KeepKind::CommandUndo] {
        assert_eq!(
            agent(ToolClass::Keep(kind), &authority, &open(), &none()),
            Verdict::Ask(AskKind::KeepAuthority)
        );
        assert_eq!(
            agent(
                ToolClass::Keep(kind),
                &path(PathClass::Normal),
                &open(),
                &none()
            ),
            Verdict::Allow(Because::Keep)
        );
    }
}

#[test]
fn every_keep_waits_for_a_running_command() {
    let running = Gates {
        command_running: true,
        ..open()
    };
    for tool in [
        ToolClass::Keep(KeepKind::Change),
        ToolClass::Keep(KeepKind::Restore),
        ToolClass::Keep(KeepKind::CommandUndo),
        ToolClass::KeepAll,
    ] {
        for class in [PathClass::Normal, PathClass::Authority] {
            assert_eq!(
                agent(tool, &path(class), &running, &none()),
                Verdict::Refuse(Reason::KeepWhileCommand),
                "{tool:?} {class:?}"
            );
        }
    }
    assert_eq!(
        Reason::KeepWhileCommand.sentence(),
        "A command is still running in this folder; Keep when it has finished."
    );
}

#[test]
fn the_lease_held_elsewhere_refuses_keeps_and_commands_but_not_reads() {
    let elsewhere = Gates {
        lease: Lease::Elsewhere,
        ..open()
    };
    for tool in [
        ToolClass::RunCommand,
        ToolClass::Keep(KeepKind::Change),
        ToolClass::Keep(KeepKind::Restore),
        ToolClass::Keep(KeepKind::CommandUndo),
        ToolClass::KeepAll,
    ] {
        assert_eq!(
            agent(tool, &command(PathClass::Normal), &elsewhere, &matched()),
            Verdict::Refuse(Reason::LeaseElsewhere),
            "{tool:?}"
        );
    }
    assert_eq!(
        agent(
            ToolClass::Read,
            &path(PathClass::Normal),
            &elsewhere,
            &none()
        ),
        Verdict::Allow(Because::Read)
    );
    // Free or held by this conversation: the action goes on (it takes the lease).
    for lease in [Lease::Free, Lease::Held] {
        let gates = Gates { lease, ..open() };
        assert_eq!(
            agent(
                ToolClass::Keep(KeepKind::Change),
                &path(PathClass::Normal),
                &gates,
                &none()
            ),
            Verdict::Allow(Because::Keep)
        );
    }
    assert_eq!(
        Reason::LeaseElsewhere.sentence(),
        "Another Lattice agent is editing this folder."
    );
}

/// The table written a second way: its rows in order, the first that matches
/// gives the verdict. `None` from a row means "does not apply".
fn table(
    mode: Mode,
    trusted: bool,
    tool: ToolClass,
    target: &Target,
    gates: &Gates,
    standing: &Standing,
) -> Verdict {
    let class = match *target {
        Target::Path(class) | Target::Command { cwd: class } => Some(class),
        Target::None => None,
    };
    let refused = match class {
        Some(PathClass::Refused(refusal)) => Some(refusal),
        _ => None,
    };
    let authority = class == Some(PathClass::Authority);
    let reads = tool == ToolClass::Read;
    let question = tool == ToolClass::AskQuestion;
    let keep = matches!(tool, ToolClass::Keep(_) | ToolClass::KeepAll);
    type Row<'a> = Box<dyn Fn() -> Option<Verdict> + 'a>;
    let rows: Vec<Row> = vec![
        Box::new(|| {
            (reads && refused.is_some()).then(|| Verdict::Refuse(Reason::Path(refused.unwrap())))
        }),
        Box::new(|| reads.then_some(Verdict::Allow(Because::Read))),
        Box::new(|| question.then_some(Verdict::Allow(Because::Question))),
        Box::new(|| (mode == Mode::Ask).then_some(Verdict::Refuse(Reason::AskMode))),
        Box::new(|| (!trusted).then_some(Verdict::Refuse(Reason::Untrusted))),
        Box::new(|| refused.map(|refusal| Verdict::Refuse(Reason::Path(refusal)))),
        Box::new(|| {
            (tool == ToolClass::Stage).then_some(Verdict::Allow(if authority {
                Because::StageAuthority
            } else {
                Because::Stage
            }))
        }),
        Box::new(|| {
            (tool == ToolClass::Mcp).then_some(if standing.mcp_allowlisted {
                Verdict::Allow(Because::Mcp)
            } else {
                Verdict::Ask(AskKind::Mcp)
            })
        }),
        Box::new(|| match tool {
            ToolClass::Browser { sensitive: true } => Some(Verdict::Ask(AskKind::Browser)),
            ToolClass::Browser { sensitive: false } => Some(Verdict::Allow(Because::Browser)),
            ToolClass::Desktop { asks: true } => Some(Verdict::Ask(AskKind::Desktop)),
            ToolClass::Desktop { asks: false } => Some(Verdict::Allow(Because::AutoMode)),
            _ => None,
        }),
        Box::new(|| {
            (gates.lease == Lease::Elsewhere).then_some(Verdict::Refuse(Reason::LeaseElsewhere))
        }),
        Box::new(|| {
            (keep && gates.command_running).then_some(Verdict::Refuse(Reason::KeepWhileCommand))
        }),
        Box::new(|| {
            (keep && authority && tool == ToolClass::KeepAll)
                .then_some(Verdict::Refuse(Reason::AuthorityKeptAlone))
        }),
        Box::new(|| (keep && authority).then_some(Verdict::Ask(AskKind::KeepAuthority))),
        Box::new(|| keep.then_some(Verdict::Allow(Because::Keep))),
        Box::new(|| {
            gates
                .command_running
                .then_some(Verdict::Refuse(Reason::CommandRunning))
        }),
        Box::new(|| {
            (gates.staged_waiting > 0).then_some(Verdict::Ask(AskKind::GateStaged {
                waiting: gates.staged_waiting,
            }))
        }),
        Box::new(|| {
            standing.command_entry.as_ref().map(|entry| {
                Verdict::Allow(Because::Standing {
                    entry: entry.clone(),
                })
            })
        }),
        Box::new(|| Some(Verdict::Ask(AskKind::Command))),
    ];
    rows.iter().find_map(|row| row()).unwrap()
}

#[test]
fn every_combination_of_inputs_agrees_with_the_table() {
    let targets = [
        Target::None,
        path(PathClass::Normal),
        path(PathClass::Authority),
        path(PathClass::Refused(PathRefusal::Ignored)),
        command(PathClass::Normal),
        command(PathClass::Refused(PathRefusal::Escape)),
    ];
    let mut checked = 0;
    let mut kinds = std::collections::BTreeSet::new();
    for mode in [Mode::Ask, Mode::Agent] {
        for trusted in [false, true] {
            for tool in TOOLS {
                for target in &targets {
                    for staged_waiting in [0, 1, 3] {
                        for command_running in [false, true] {
                            for lease in [Lease::Free, Lease::Held, Lease::Elsewhere] {
                                for standing in [
                                    none(),
                                    matched(),
                                    Standing {
                                        command_entry: None,
                                        mcp_allowlisted: true,
                                    },
                                ] {
                                    let gates = Gates {
                                        staged_waiting,
                                        command_running,
                                        lease,
                                    };
                                    let got =
                                        decide(mode, trusted, tool, target, &gates, &standing);
                                    let want =
                                        table(mode, trusted, tool, target, &gates, &standing);
                                    assert_eq!(
                                        got, want,
                                        "{mode:?} trusted={trusted} {tool:?} {target:?} {gates:?} {standing:?}"
                                    );
                                    kinds.insert(got.rule());
                                    checked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 2 * 2 * 13 * 6 * 3 * 2 * 3 * 3);
    // Every rule of the engine was reached.
    assert_eq!(kinds.len(), 22, "{kinds:?}");
}
