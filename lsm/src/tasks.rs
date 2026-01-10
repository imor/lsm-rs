//! Background task management for LSM database operations.
//!
//! This module provides a flexible task management system for coordinating background
//! operations in the LSM-tree database, primarily focused on compaction tasks.
//!
//! # Architecture
//!
//! The task system consists of several key components:
//! - **TaskManager**: Central coordinator for all background tasks
//! - **TaskHandle**: Individual task execution wrapper with lifecycle management
//! - **TaskGroup**: Groups related tasks (e.g., all compaction tasks)
//! - **UpdateCond**: Condition variable for efficient task wake-up
//!
//! # Task Types
//!
//! Currently supports two types of compaction tasks:
//! - **Memtable Compaction**: Flushes in-memory memtables to L0 SSTables
//! - **Level Compaction**: Merges SSTables between levels
//!
//! # Graceful Shutdown
//!
//! The system supports graceful shutdown via atomic flags. When `stop_all()` is called,
//! all tasks complete their current work and then terminate cleanly.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use parking_lot::RwLock;

use tokio::sync::Notify;

use crate::Error;
use crate::logic::DbLogic;

use async_trait::async_trait;

/// A trait for background tasks that can be executed by the task manager.
///
/// Tasks are long-running operations that execute in the background and can be
/// started, stopped, and coordinated with other tasks.
#[async_trait]
pub trait Task: Sync + Send {
    /// Execute one iteration of the task's work.
    ///
    /// Returns `Ok(true)` if work was performed, `Ok(false)` if there was no work to do.
    /// This return value is used to determine whether the task should wait for new work
    /// or immediately try again.
    async fn run(&self) -> Result<bool, Error>;
}

/// Identifies the type of background task.
///
/// Used to distinguish between different categories of background work and
/// to wake up specific task groups.
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Tasks that flush memtables to L0 SSTables on disk.
    MemtableFlush,
    /// Tasks that compact SSTables between different levels.
    LevelCompaction,
}

/// A handle to a running background task.
///
/// Manages the lifecycle of a single task, including its stop flag,
/// the task implementation, and the condition variable used to wake it up.
struct TaskHandle {
    /// Shared flag indicating whether the task should stop.
    stop_flag: Arc<AtomicBool>,
    /// The actual task implementation.
    task: Box<dyn Task>,
    /// Condition variable for efficient wake-up when new work is available.
    update_cond: Arc<UpdateCond>,
}

/// Manages all background tasks for the database.
///
/// The TaskManager is responsible for:
/// - Creating and spawning background task workers
/// - Coordinating task wake-ups when new work is available
/// - Managing graceful shutdown of all tasks
///
/// Currently manages compaction tasks, but is designed to be extensible
/// for additional background operations in the future.
pub struct TaskManager {
    /// Global stop flag shared by all tasks.
    stop_flag: Arc<AtomicBool>,
    /// Map of task types to their corresponding task groups.
    task_groups: HashMap<TaskType, TaskGroup>,
}

/// A group of tasks that perform the same type of work.
///
/// For example, all level compaction tasks belong to the same group
/// and share a condition variable for wake-up coordination.
struct TaskGroup {
    /// Shared condition variable for all tasks in this group.
    condition: Arc<UpdateCond>,
}

/// Condition variable with timestamp tracking for task coordination.
///
/// Tracks when work was last made available and provides efficient
/// wake-up notifications to waiting tasks.
struct UpdateCond {
    /// Timestamp of the last change that made work available.
    last_change: RwLock<Instant>,
    /// Tokio notify primitive for waking up waiting tasks.
    condition: Notify,
}

/// Task that handles flushing memtables to L0 SSTables.
///
/// When a memtable is frozen, this task wakes up
/// the level compaction tasks since new L0 tables may trigger compactions.
struct MemtableFlushTask {
    /// Reference to the database logic layer.
    dblogic: Arc<DbLogic>,
    /// Condition variable to wake up level compaction tasks.
    level_update_cond: Arc<UpdateCond>,
}

/// Task that handles compacting SSTables between levels.
///
/// Performs the bulk of compaction work, merging and rewriting SSTables
/// to maintain the LSM-tree invariants and reclaim space.
struct LevelCompactionTask {
    /// Reference to the database logic layer.
    datastore: Arc<DbLogic>,
}

impl MemtableFlushTask {
    /// Creates a new boxed memtable compaction task.
    ///
    /// # Arguments
    /// * `datastore` - Reference to the database logic layer
    /// * `level_update_cond` - Condition variable to notify level compaction tasks
    fn new_boxed(dblogic: Arc<DbLogic>, level_update_cond: Arc<UpdateCond>) -> Box<dyn Task> {
        Box::new(Self {
            dblogic,
            level_update_cond,
        })
    }
}

impl LevelCompactionTask {
    /// Creates a new boxed level compaction task.
    ///
    /// # Arguments
    /// * `datastore` - Reference to the database logic layer
    fn new_boxed(datastore: Arc<DbLogic>) -> Box<dyn Task> {
        Box::new(Self { datastore })
    }
}

#[async_trait]
impl Task for MemtableFlushTask {
    /// Executes one memtable compaction operation.
    ///
    /// If a memtable was successfully flushed, wakes up level compaction tasks
    /// since new L0 tables may have been created.
    async fn run(&self) -> Result<bool, Error> {
        let did_work = self.dblogic.flush_frozen_memtable().await?;
        if did_work {
            self.level_update_cond.wake_up();
        }
        Ok(did_work)
    }
}

#[async_trait]
impl Task for LevelCompactionTask {
    /// Executes one level compaction operation.
    ///
    /// Selects and compacts SSTables between levels according to the
    /// compaction strategy.
    async fn run(&self) -> Result<bool, Error> {
        Ok(self.datastore.do_level_compaction().await?)
    }
}

impl UpdateCond {
    /// Creates a new condition variable with current timestamp.
    fn new() -> Self {
        Self {
            last_change: RwLock::new(Instant::now()),
            condition: Default::default(),
        }
    }

    /// Notifies waiting tasks that there is new work available.
    ///
    /// Updates the timestamp and wakes up one waiting task. The task
    /// will check if the timestamp is newer than its last processed work.
    fn wake_up(&self) {
        let mut last_change = self.last_change.write();
        *last_change = Instant::now();
        self.condition.notify_one();
    }
}

impl TaskHandle {
    /// Creates a new task handle.
    ///
    /// # Arguments
    /// * `stop_flag` - Shared flag for stopping the task
    /// * `update_cond` - Condition variable for wake-up notifications
    /// * `task` - The task implementation to execute
    fn new(stop_flag: Arc<AtomicBool>, update_cond: Arc<UpdateCond>, task: Box<dyn Task>) -> Self {
        Self {
            stop_flag,
            update_cond,
            task,
        }
    }

    /// Checks if the task should continue running.
    ///
    /// Returns `true` if the stop flag has not been set.
    #[inline(always)]
    fn stop_requested(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }

    /// Main work loop for the task.
    ///
    /// Continuously executes the task until the stop flag is set. When idle
    /// (no work available), waits on the condition variable for new work.
    /// The waiting is efficient and only wakes up when:
    /// - New work is explicitly signaled via wake_up()
    /// - The stop flag is set
    async fn work_loop(&self) {
        log::trace!("Task work loop started");
        let mut last_update = Instant::now();

        // Indicates whether work was done in the last iteration
        let mut idle = false;

        loop {
            let now = Instant::now();

            loop {
                let fut = self.update_cond.condition.notified();
                tokio::pin!(fut);

                {
                    let last_change = self.update_cond.last_change.read();

                    if self.stop_requested() || !idle || *last_change > last_update {
                        break;
                    }

                    // wait for change to queue and retry
                    fut.as_mut().enable();
                }

                fut.await;
            }

            if self.stop_requested() {
                break;
            }

            let did_work = self.task.run().await.expect("Task failed");
            last_update = now;

            if did_work {
                idle = false;
            } else {
                log::trace!("Task did not do any work");
                idle = true;
            }
        }

        log::trace!("Task work loop ended");
    }
}

impl TaskManager {
    /// Creates a new task manager and spawns background workers.
    ///
    /// # Arguments
    /// * `datastore` - Reference to the database logic layer
    /// * `num_compaction_tasks` - Number of level compaction worker threads to spawn
    ///
    /// Spawns one memtable compaction task and the specified number of level
    /// compaction tasks. All tasks start immediately and begin waiting for work.
    pub async fn new(dblogic: Arc<DbLogic>, num_compaction_tasks: usize) -> Self {
        let mut task_groups = HashMap::default();
        let stop_flag = Arc::new(AtomicBool::new(false));

        let memtable_update_cond = Arc::new(UpdateCond::new());
        let level_update_cond = Arc::new(UpdateCond::new());

        // Spawn memtable flush task
        {
            let stop_flag = stop_flag.clone();
            let memtable_update_cond = memtable_update_cond.clone();
            let level_update_cond = level_update_cond.clone();
            let dblogic = dblogic.clone();

            let memtable_flush_task_handle = TaskHandle::new(
                stop_flag,
                memtable_update_cond,
                MemtableFlushTask::new_boxed(dblogic, level_update_cond),
            );

            tokio::spawn(async move { memtable_flush_task_handle.work_loop().await });
        }

        let task_group = TaskGroup {
            condition: memtable_update_cond,
        };

        task_groups.insert(TaskType::MemtableFlush, task_group);

        // Spawn level compaction tasks
        {
            for _ in 0..num_compaction_tasks {
                let stop_flag = stop_flag.clone();
                let level_update_cond = level_update_cond.clone();
                let dblogic = dblogic.clone();

                let level_compaction_task_handle = TaskHandle::new(
                    stop_flag,
                    level_update_cond,
                    LevelCompactionTask::new_boxed(dblogic),
                );

                tokio::spawn(async move { level_compaction_task_handle.work_loop().await });
            }

            let task_group = TaskGroup {
                condition: level_update_cond,
            };

            task_groups.insert(TaskType::LevelCompaction, task_group);
        }

        Self {
            stop_flag,
            task_groups,
        }
    }

    /// Wakes up tasks of the specified type.
    ///
    /// Signals to tasks in the specified task group that new work is available.
    /// One task from the group will wake up and check for work.
    ///
    /// # Arguments
    /// * `task_type` - The type of tasks to wake up
    ///
    /// # Panics
    /// Panics if the task type is not registered.
    #[tracing::instrument(skip(self))]
    pub fn wake_up(&self, task_type: &TaskType) {
        let task_group = self.task_groups.get(task_type).expect("No such task group");
        task_group.condition.wake_up();
    }

    /// Terminates all tasks immediately (legacy method).
    pub fn terminate(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);

        for (_, task_group) in self.task_groups.iter() {
            task_group.condition.condition.notify_one();
        }
    }

    /// Gracefully stops all background tasks.
    ///
    /// Sets the stop flag and wakes up all waiting tasks. Tasks will complete
    /// their current work iteration and then exit their work loops.
    ///
    /// This method returns immediately after signaling; it does not wait for
    /// tasks to actually terminate. The tasks are spawned with `tokio::spawn`
    /// and will clean up asynchronously.
    pub async fn stop_all(&self) -> Result<(), Error> {
        log::trace!("Stopping all background tasks");

        self.stop_flag.store(true, Ordering::SeqCst);

        for (_, task_group) in self.task_groups.iter() {
            task_group.condition.condition.notify_waiters();
        }

        Ok(())
    }
}
